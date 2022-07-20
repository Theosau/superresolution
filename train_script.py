import torch, tqdm
from cnn_models import ConvAE
from loss_functions import PDELoss
from torch.utils.data import DataLoader, Dataset

class ChannelFLow(Dataset):
    def __init__(self, data):
        super(ChannelFLow, self).__init__()
        self.data = data
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        self.slice = self.data[idx]
        return self.slice

if __name__ == "__main__":

    # setting device and data types
    dtype = torch.float32 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # image size
    input_size = 320

    # beta loss weight parameter
    beta = 0.5

    # epochs
    num_epochs = 10

    # willl have to load the data
    x = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    y = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    u = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    v = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    p = torch.randn((10, input_size, input_size, 1), requires_grad=True)

    x_val = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    y_val = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    u_val = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    v_val = torch.randn((10, input_size, input_size, 1), requires_grad=True)
    p_val = torch.randn((10, input_size, input_size, 1), requires_grad=True)

    # data
    train_data = torch.cat([x, y, u, v, p], axis=-1)
    train_data = train_data.permute((0, 3, 1, 2))
    val_data = torch.cat([x_val, y_val, u_val, v_val, p_val], axis=-1)
    val_data = val_data.permute((0, 3, 1, 2)) 

    # datasets
    train_dataset = ChannelFLow(train_data)
    val_dataset = ChannelFLow(val_data)

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # setup model
    model = ConvAE(input_size=input_size)
    model = model.to(device=device)
    model.train()

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ae_loss_function = torch.nn.MSELoss(reduction='mean')
    pde_loss_function = PDELoss()
            
    for epoch in tqdm(range(num_epochs)):
        for i, slice in tqdm(enumerate(train_dataloader)):
            slice = slice.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            # ===================forward=====================
            reconstruction = model.forward(slice)
            # =====================loss======================
            ae_loss = ae_loss_function(slice[:, 2:], reconstruction)
            pde_loss = pde_loss_function(slice[:, 0], slice[:, 1], slice[:, 2], slice[:, 3], slice[:, 4])
            loss = beta * ae_loss + (1-beta) * pde_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()