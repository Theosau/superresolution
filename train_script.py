import torch, pdb, os, yaml
import numpy as np
from tqdm import tqdm
from cnn_models import ConvAE, ConvUNet
from loss_functions import PDELoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms.functional import normalize

class ChannelFlow(Dataset):
    def __init__(self, dataset='train', data_path='example/'):
        super(ChannelFlow, self).__init__()
        self.dataset= dataset
        self.data_path = data_path
        return
    
    def __len__(self):
        return len(os.listdir('data/' + self.data_path + self.dataset +'/')) // 3

    def __getitem__(self, idx):
        req_grad = True if self.dataset=='train' else False 
        self.x = torch.tensor(np.load('data/' + self.data_path + self.dataset + f'/xs_sample{idx}.npy'), requires_grad=req_grad).unsqueeze(dim=0)
        self.y = torch.tensor(np.load('data/' + self.data_path + self.dataset + f'/ys_sample{idx}.npy'), requires_grad=req_grad).unsqueeze(dim=0)
        self.slice = torch.tensor(np.load(f'data/' + self.data_path + self.dataset + f'/sample{idx}.npy'), requires_grad=req_grad).unsqueeze(dim=0)

        # interpolation
        self.x = torch.nn.functional.interpolate(self.x, scale_factor=2, mode='bilinear').squeeze(dim=0)
        self.y = torch.nn.functional.interpolate(self.y, scale_factor=2, mode='bilinear').squeeze(dim=0)
        self.slice = torch.nn.functional.interpolate(self.slice, scale_factor=2, mode='bilinear').squeeze(dim=0)

        return self.x, self.y, self.slice

if __name__ == "__main__":

    # loading the model config
    with open('training_config.yml') as file:
        config = yaml.safe_load(file)

    # setting device and data types
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # image size
    reshape_size = config['reshape_size']
    data_path = config['data_path']

    # reconstruction loss weight parameter
    beta = config['beta']
    # pde loss weight parameter
    alpha = config['beta']

    # epochs
    num_epochs = config['num_epochs']

    # model name
    model_name = config['model_name']

    # datasets
    train_dataset = ChannelFlow(dataset='train', data_path=data_path)
    val_dataset = ChannelFlow(dataset='val', data_path=data_path)

    # dataloaders
    batch_size = config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # setup model
    model = ConvUNet(input_size=reshape_size, channels_init=config['channels_init'])
    if 'continued' in model_name:
        checkpoint = torch.load('trainings/saved_models/pde_only_cpu_scheduler_batch8')
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device)
    model.train()
    print(sum(p.numel() for p in model.parameters()))

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    if 'continued' in model_name:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    scheduler = ExponentialLR(optimizer, config['scheduler_gamma'])
    recon_loss_function = torch.nn.MSELoss(reduction='mean')
    pde_loss_function = PDELoss()

    # setup the writer
    writer = SummaryWriter(log_dir='trainings/logs/' + model_name)
    write_counter = 0
    if 'continued' in model_name:
        write_counter = checkpoint['write_counter'] 
    
    # initialize iterations loss to 0
    its_loss = 0
    its_pde_loss = 0
    its_recon_loss = 0
    its_boundary_loss = 0

    print('Setup the model, dataloader, datasets, loss funcitons, optimizers.')     
    for epoch in tqdm(range(num_epochs)):

        # iterate through the dataset
        for i, (x_sample, y_sample, flow_sample) in enumerate(train_dataloader):

            x_sample = x_sample.to(device=device, dtype=dtype)
            y_sample = y_sample.to(device=device, dtype=dtype)
            flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU

            x_sample_norm = normalize(x_sample, (0), (1))
            y_sample_norm = normalize(y_sample, (0), (1))

            means = flow_sample.mean(dim=(-1, -2), keepdim=True)
            stds = flow_sample.std(dim=(-1, -2), keepdim=True)
            flow_sample_norm = (flow_sample - means) / stds

            # ===================forward=====================
            reconstruction_norm  = model.forward(torch.cat([x_sample_norm, y_sample_norm, flow_sample_norm], axis=1))
            reconstruction = reconstruction_norm * stds + means

            # =====================losses======================
            # apply boundary conditions loss - for now Dirichlet assuming boundary are noiseless
            boundary_loss = recon_loss_function(reconstruction_norm[:, 0:2, 0, :], flow_sample_norm[:, 0:2, 0, :]) + \
                            recon_loss_function(reconstruction_norm[:, 0:2, -1, :], flow_sample_norm[:, 0:2, -1, :]) + \
                            recon_loss_function(reconstruction_norm[:, 0:2, :, 0], flow_sample_norm[:, 0:2, :, 0]) + \
                            recon_loss_function(reconstruction_norm[:, 0:2, :, -1], flow_sample_norm[:, 0:2, :, -1])
            # PDE loss
            if beta < 1:
                pde_loss = pde_loss_function.compute_loss(x_sample, y_sample, reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2])
            else:
                pde_loss = 0
            # reconstruction loss
            if beta > 0:
                recon_loss = recon_loss_function(flow_sample_norm, reconstruction_norm)
            else:
                recon_loss = 0
            loss = beta * recon_loss + (1-beta) * (alpha * pde_loss + (1 - alpha) * boundary_loss)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                scheduler.step()

            # update write iteration loss
            its_loss += loss
            its_pde_loss += pde_loss
            its_recon_loss += recon_loss
            its_boundary_loss += boundary_loss

            if (i+1) % 10:
                np.save('reconstuction.npy', reconstruction[0, 0].detach().cpu().numpy())
                # write to tensorboard
                writer.add_scalar("Loss/train", its_loss/10, write_counter)
                writer.add_scalar("Reconstruction Loss/train", its_recon_loss/10, write_counter)
                writer.add_scalar("PDE Loss/train", its_pde_loss/10, write_counter)
                writer.add_scalar("Bonudary Loss/train", its_boundary_loss/10, write_counter)
                write_counter += 1

                # reset losses to 0
                its_loss = 0
                its_pde_loss = 0
                its_recon_loss = 0
                its_boundary_loss = 0

                # plot images
                img_batch = torch.zeros((2, 1, reshape_size, reshape_size))
                img_batch[0, 0] = flow_sample[0, 0, ...]
                img_batch[1, 0] = reconstruction[0, 0, ...]
                writer.add_images('example reconstruction', img_batch, 0, dataformats='NCHW')

        # save model
        if (epoch+1) % 10:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'write_counter': write_counter,
                        'config': config,
                        }, 'trainings/saved_models/' + model_name)
            print(f'Saved model, epoch {i}.')