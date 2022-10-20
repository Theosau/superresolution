import torch
import numpy as np
from tqdm import tqdm
from helper_functions import make_blocks_vectorized

# file 
file_name = 'channel_x1_256_y1_512_z1_256_step2'

# load data
x_all = np.load('data/' + file_name + '/xs.npy')
y_all = np.load('data/' + file_name + '/ys.npy')
u_all = np.load('data/' + file_name + '/us.npy')
v_all = np.load('data/' + file_name + '/vs.npy')
p_all = np.load('data/' + file_name + '/ps.npy')

reshape_size = 8

x_all = np.reshape(make_blocks_vectorized(x_all, reshape_size), (-1, reshape_size, reshape_size))
y_all = np.reshape(make_blocks_vectorized(y_all, reshape_size), (-1, reshape_size, reshape_size))
u_all = np.reshape(make_blocks_vectorized(u_all, reshape_size), (-1, reshape_size, reshape_size))
v_all = np.reshape(make_blocks_vectorized(v_all, reshape_size), (-1, reshape_size, reshape_size))
p_all = np.reshape(make_blocks_vectorized(p_all, reshape_size), (-1, reshape_size, reshape_size))

len_train = int(0.8*len(x_all))

x = torch.tensor(x_all[:len_train, :, :], requires_grad=True).unsqueeze(-1).permute((0, 3, 1, 2))
y = torch.tensor(y_all[:len_train, :, :], requires_grad=True).unsqueeze(-1).permute((0, 3, 1, 2))
u = torch.tensor(u_all[:len_train, :, :], requires_grad=True).unsqueeze(-1)
v = torch.tensor(v_all[:len_train, :, :], requires_grad=True).unsqueeze(-1)
p = torch.tensor(p_all[:len_train, :, :], requires_grad=True).unsqueeze(-1)

x_val = torch.tensor(x_all[len_train:, :, :], requires_grad=True).unsqueeze(-1).permute((0, 3, 1, 2))
y_val = torch.tensor(y_all[len_train:, :, :], requires_grad=True).unsqueeze(-1).permute((0, 3, 1, 2))
u_val = torch.tensor(u_all[len_train:, :, :], requires_grad=True).unsqueeze(-1)
v_val = torch.tensor(v_all[len_train:, :, :], requires_grad=True).unsqueeze(-1)
p_val = torch.tensor(p_all[len_train:, :, :], requires_grad=True).unsqueeze(-1)

# data
train_data = torch.cat([u, v, p], axis=-1)
train_data = train_data.permute((0, 3, 1, 2))
val_data = torch.cat([u_val, v_val, p_val], axis=-1)
val_data = val_data.permute((0, 3, 1, 2)) 

# convert to numpy and save
train_x_np = x.detach().numpy()
train_y_np = y.detach().numpy()
train_data_np = train_data.detach().numpy()
val_x_np = x_val.detach().numpy()
val_y_np = y_val.detach().numpy()
val_data_np = val_data.detach().numpy()

for i in tqdm(range(len(train_data_np))):
    np.save(f'data/example8/train/sample{i}.npy', train_data_np[i])
    np.save(f'data/example8/train/xs_sample{i}.npy', train_x_np[i])
    np.save(f'data/example8/train/ys_sample{i}.npy', train_y_np[i])

for i in tqdm(range(len(val_data_np))):
    np.save(f'data/example8/val/sample{i}.npy', val_data_np[i])
    np.save(f'data/example8/val/xs_sample{i}.npy', val_x_np[i])
    np.save(f'data/example8/val/ys_sample{i}.npy', val_y_np[i])