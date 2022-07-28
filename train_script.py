import torch, pdb
from tqdm import tqdm
from cnn_models import ConvAE
from loss_functions import PDELoss
from torch.utils.data import DataLoader, Dataset

class ChannelFLow(Dataset):
    def __init__(self, x, y, data):
        super(ChannelFLow, self).__init__()
        self.x = x
        self.y = y
        self.data = data
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        self.x = x[idx]
        self.y = y[idx]
        self.slice = self.data[idx]
        return self.x, self.y, self.slice

if __name__ == "__main__":

    # setting device and data types
    dtype = torch.float32 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # image size
    input_size = 320
    num_samples = 32

    # beta loss weight parameter
    beta = 0.

    # epochs
    num_epochs = 10

    # willl have to load the data
    print('Generating random data')
    x = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True).permute((0, 3, 1, 2))
    y = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True).permute((0, 3, 1, 2))
    u = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True)
    v = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True)
    p = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True)

    x_val = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True).permute((0, 3, 1, 2))
    y_val = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True).permute((0, 3, 1, 2))
    u_val = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True)
    v_val = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True)
    p_val = torch.randn((num_samples, input_size, input_size, 1), requires_grad=True)
    print('Generated data')
    
    # data
    train_data = torch.cat([u, v, p], axis=-1)
    train_data = train_data.permute((0, 3, 1, 2))
    val_data = torch.cat([u_val, v_val, p_val], axis=-1)
    val_data = val_data.permute((0, 3, 1, 2)) 

    # pdb.set_trace()
    # datasets
    train_dataset = ChannelFLow(x, y, train_data)
    val_dataset = ChannelFLow(x_val, y_val, val_data)

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # setup model
    model = ConvAE(input_size=input_size)
    model = model.to(device=device)
    model.train()

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ae_loss_function = torch.nn.MSELoss(reduction='mean')
    pde_loss_function = PDELoss()

    print('Setup the model, dataloader, datasets, loss funcitons, optimizers.')     
    for epoch in tqdm(range(num_epochs)):
        for i, (x_sample, y_sample, slice) in tqdm(enumerate(train_dataloader)):
            x_sample = x_sample.to(device=device, dtype=dtype)
            y_sample = y_sample.to(device=device, dtype=dtype)
            slice = slice.to(device=device, dtype=dtype) #.requires_grad_(True)  # move to device, e.g. GPU
            # ===================forward=====================
            reconstruction = model.forward(torch.cat([x_sample, y_sample, slice], axis=1))
            # =====================loss======================
            pde_loss = pde_loss_function.compute_loss(x_sample, y_sample, reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2])
            ae_loss = ae_loss_function(slice, reconstruction)
            loss = beta * ae_loss + (1-beta) * pde_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)