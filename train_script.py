import torch, pdb, os, yaml
import numpy as np
from tqdm import tqdm
from cnn_models import ConvUNetTwoModels, SmallLinear
from loss_functions import PDELoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms.functional import normalize
from torch.nn.functional import interpolate

# custom dataset
class ChannelFlowFull(Dataset):
    def __init__(self, x, y, vels, seg_map):
        super(ChannelFlowFull, self).__init__()
        self.x = x
        self.y = y
        self.seg_map = seg_map
        self.vels = vels
        return
    
    def __len__(self):
        return len(self.vels)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.vels[idx], self.seg_map[idx]

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
    scale_factor = config['scale_factor']
    file_name = 'channel_x1_256_y1_512_z1_256_step2'

    # reconstruction loss weight parameter
    beta = config['beta']
    # pde loss weight parameter
    alpha = config['beta']

    # epochs
    num_epochs = config['num_epochs']

    # model name
    model_name = config['model_name']
    final_out_channels = config['final_out_channels']

    # load data
    x_all = np.load('data/' + file_name + '/xs.npy')
    y_all = np.load('data/' + file_name + '/ys.npy')
    u_all = np.load('data/' + file_name + '/us.npy')
    v_all = np.load('data/' + file_name + '/vs.npy')
    p_all = np.load('data/' + file_name + '/ps.npy')

    # segmentation maps
    seg_map_all = 2*np.ones_like(x_all)
    seg_map_all[:, 0, :] = 1
    seg_map_all[:, -1, :] = 1

    # initial datasets
    len_train = int(0.8*len(x_all))
    x, y, seg_map = [
        torch.tensor(arr[:len_train, :, :], requires_grad=True).unsqueeze(-1).permute((0, 3, 1, 2)) for arr in [x_all, y_all, seg_map_all]
    ]
    u, v, p = [
        torch.tensor(arr[:len_train, :, :], requires_grad=True).unsqueeze(-1) for arr in [u_all, v_all, p_all]
    ]
    x_val, y_val, seg_map_val = [
        torch.tensor(arr[len_train:, :, :], requires_grad=False).unsqueeze(-1).permute((0, 3, 1, 2)) for arr in [x_all, y_all, seg_map_all]
    ]
    u_val, v_val, p_val = [
        torch.tensor(arr[len_train:, :, :], requires_grad=False).unsqueeze(-1) for arr in [u_all, v_all, p_all]
    ]

    # split data
    train_data = torch.cat([u, v, p], axis=-1)
    train_data = train_data.permute((0, 3, 1, 2))
    val_data = torch.cat([u_val, v_val, p_val], axis=-1)
    val_data = val_data.permute((0, 3, 1, 2)) 

    # datasets
    train_dataset = ChannelFlowFull(x, y, train_data, seg_map)
    val_dataset = ChannelFlowFull(x_val, y_val, val_data, seg_map_val)

    # dataloaders
    batch_size = config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # setup model
    cnn_model = ConvUNetTwoModels(input_size=256, channels_init=4, final_out_channels=final_out_channels)
    cnn_model = cnn_model.to(device=device)
    cnn_model.train()

    linear_model = SmallLinear(input_features=final_out_channels+2) # + 2 is for the u v output from the model 
    # (won't have p on 4D flow images)
    linear_model = linear_model.to(device=device)
    linear_model.train()
    
    all_params = list(cnn_model.parameters()) + list(linear_model.parameters())
    num_cnn_params = sum(p.numel() for p in cnn_model.parameters())
    num_linear_params = sum(p.numel() for p in linear_model.parameters())
    print('There are ', num_cnn_params+num_linear_params, ' parameters to train.')

    # Train the model
    optimizer = torch.optim.Adam(all_params, lr=config['learning_rate'])
    # if 'continued' in model_name:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    scheduler = ExponentialLR(optimizer, config['scheduler_gamma'])
    recon_loss_function = torch.nn.MSELoss(reduction='mean')
    pde_loss_function = PDELoss()

    # setup the writer
    writer = SummaryWriter(log_dir='trainings/logs/' + model_name)
    write_counter = 0
    # if 'continued' in model_name:
    #     write_counter = checkpoint['write_counter'] 

    # initialize iterations loss to 0
    its_loss = 0
    its_pde_loss = 0
    its_recon_loss = 0
    its_boundary_loss = 0

    print('Setup the model, dataloader, datasets, loss funcitons, optimizers.')     
    for epoch in tqdm(range(num_epochs)):

        # iterate through the dataset
        for i, (x_sample, y_sample, flow_sample, seg_map_sample) in enumerate(train_dataloader):

            # make sure there is no gradient
            optimizer.zero_grad()

            x_sample = x_sample.to(device=device, dtype=dtype)
            y_sample = y_sample.to(device=device, dtype=dtype)
            flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU
            seg_map_sample = seg_map_sample.to(device=device, dtype=dtype).squeeze()
    
            # =====================forward======================

            # compute latent vectors
            latent_vectors = cnn_model(flow_sample)

            # separate the points
            latent_vectors_perm = latent_vectors.permute(0, 2, 3, 1)
            interior_points = latent_vectors_perm[seg_map_sample==2] # all interior points from the batch
            boundary_points = latent_vectors_perm[seg_map_sample==1] # all boundary points from the batch

            # permute to easily access the desired points via indexing
            x_sample_perm = x_sample.permute(0, 2, 3, 1)
            y_sample_perm = y_sample.permute(0, 2, 3, 1)
            flow_sample_perm = flow_sample.permute(0, 2, 3, 1)

            # group the points
            x_interior_points = x_sample_perm[seg_map_sample==2].view(-1, 1)
            x_boundary_points = x_sample_perm[seg_map_sample==1].view(-1, 1)

            y_interior_points = y_sample_perm[seg_map_sample==2].view(-1, 1)
            y_boundary_points = y_sample_perm[seg_map_sample==1].view(-1, 1)

            labels_interior_points = flow_sample_perm[seg_map_sample==2].view(-1, 3)
            labels_boundary_points = flow_sample_perm[seg_map_sample==1].view(-1, 3)

            # split input features to allow taking separate derivatives
            input_features_interior = torch.cat([x_interior_points, y_interior_points, interior_points], dim=1)
            input_features_boundary = torch.cat([x_boundary_points, y_boundary_points, boundary_points], dim=1)
            input_features = torch.cat([input_features_interior, input_features_boundary], dim=0)
            
            latent_inputs = [input_features[..., i:i+1] for i in range(input_features.shape[-1])]
            x_ = torch.cat(latent_inputs, axis=-1)

            # forward through linear model
            outputs_linear = linear_model(x_)

            # inputs_interior = [input_features_interior[..., i:i+1] for i in range(input_features_interior.shape[-1])]
            # latent_inputs_interior = torch.cat(inputs_interior, axis=-1)

            # input_features_boundary = torch.cat([x_boundary_points, y_boundary_points, boundary_points], dim=1)
            # inputs_boundary = [input_features_boundary[..., i:i+1] for i in range(input_features_boundary.shape[-1])]
            # latent_inputs_boundary = torch.cat(inputs_boundary, axis=-1)

            # latent_inputs = torch.cat([latent_inputs_interior, latent_inputs_boundary], dim=0)
            # =====================losses======================
            # PDE loss for interior points
            if beta < 1:
                pde_loss = pde_loss_function.compute_loss(outputs_linear[:len(input_features_interior)], latent_inputs[:2])
            else:
                pde_loss = 0
            # reconstruction loss for interior
            if beta > 0:
                # pdb.set_trace()
                recon_loss = recon_loss_function(outputs_linear[:len(input_features_interior)], labels_interior_points)
            else:
                recon_loss = 0
            # reconstruction loss for boundary
            boundary_loss = recon_loss_function(outputs_linear[len(input_features_interior):], labels_boundary_points)

            # total loss
            loss = beta * recon_loss + (1-beta) * (alpha * pde_loss + (1 - alpha) * boundary_loss)
            print(loss.item())

            # ===================backward====================
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                scheduler.step()

            # update write iteration loss
            its_loss += loss
            its_pde_loss += pde_loss
            its_recon_loss += recon_loss
            its_boundary_loss += boundary_loss

            # if (i+1) % 10:
            #     np.save('reconstuction.npy', reconstruction[0, 0].detach().cpu().numpy())
            #     # write to tensorboard
            #     writer.add_scalar("Loss/train", its_loss/10, write_counter)
            #     writer.add_scalar("Reconstruction Loss/train", its_recon_loss/10, write_counter)
            #     writer.add_scalar("PDE Loss/train", its_pde_loss/10, write_counter)
            #     writer.add_scalar("Bonudary Loss/train", its_boundary_loss/10, write_counter)
            #     write_counter += 1

            #     # reset losses to 0
            #     its_loss = 0
            #     its_pde_loss = 0
            #     its_recon_loss = 0
            #     its_boundary_loss = 0

            #     # plot images
            #     img_batch = torch.zeros((2, 1, reshape_size//2, reshape_size//2))
            #     img_batch[0, 0] = flow_sample[0, 0, ...]
            #     img_batch[1, 0] = reconstruction[0, 0, ...]
            #     writer.add_images('example reconstruction', img_batch, 0, dataformats='NCHW')

        # save model
        # if (epoch+1) % 10:
        #     torch.save({
        #                 'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'write_counter': write_counter,
        #                 'config': config,
        #                 }, 'trainings/saved_models/' + model_name)
        #     print(f'Saved model, epoch {i}.')