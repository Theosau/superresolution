import torch, pdb, os, yaml, shutil
import numpy as np
from tqdm import tqdm
from geometry_models import ConvNet
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from torch.nn import MSELoss, L1Loss
from torch.cuda.amp import GradScaler 
from torch import autocast

import sys
sys.path.append('/notebooks/superresolution/')
curdir = os.getcwd()
os.chdir('..')
print(os.getcwd())
from data.data_generation import generate_dataset, generate_pipe_flow_dataset, generate_3dstenosis, generate_3daneurysm
os.chdir(curdir)

# Custom dataset
class PoseuilleFlowSDF(Dataset):
    def __init__(self, sdf):
        super(PoseuilleFlowSDF, self).__init__()
        self.sdf = sdf
        return

    def __len__(self):
        return len(self.sdf)

    def __getitem__(self, idx):
        return self.sdf[idx]

if __name__ == "__main__":

    # loading the model config
    with open("geometry_training_config.yml") as file:
        config = yaml.safe_load(file)

    # model name
    model_name = config["model_name"]

    # save config file for reload
    shutil.copyfile("geometry_training_config.yml", f"geometry_trainings/geometry_config_models/{model_name}.yml")
    shutil.copyfile("geometry_models.py", f"geometry_trainings/geometry_models/{model_name}.py")

    # setting device and data types
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # volume size
    nvox = config["nvox"]
    # nsamples = config["nsamples"]
    # multthickness = config["multthickness"]
    # multpositions = config["multpositions"]
    # multangles = config["multangles"]
    # flows_in_z = config["flows_in_z"]
    epoch_sparsity = config["epoch_sparsity"]
    target_sparsity = config["target_sparsity"]

    # datasets
    # gen_channel = config["gen_channel"]
    # gen_pipe = config["gen_pipe"]
    # assert gen_channel or gen_pipe, 'You should at least generate channel flows or pipe flows.'
    gen_stenosis = config["gen_stenosis"]
    gen_aneurysm = config["gen_aneurysm"]
    assert gen_stenosis or gen_aneurysm, 'You should at least generate stenosis flows or aneurysm flows.'
    
    # augmentation
    transforms = config["transforms"]
    data_aug = config["data_aug"]

    if gen_stenosis:
        _, _, _, sdfs, _ = generate_3dstenosis(
            nvox=64, 
            sdf=True,
            transforms=transforms,
            data_aug=data_aug
        )

    if gen_aneurysm:
        _, _, _, sdfs_aneurysm, _ = generate_3daneurysm(
            nvox=64, 
            sdf=True,
            transforms=transforms,
            data_aug=data_aug
        )

    if gen_stenosis and gen_aneurysm:
        # concatenate sdfs all together
        sdfs = np.concatenate((sdfs, sdfs_aneurysm), axis=0)
        del sdfs_aneurysm
    # just pipe flows
    elif gen_aneurysm:
        sdfs = sdfs_aneurysm

    # normalize the signed distance fields
    sdfs = (sdfs - np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True))/(np.max(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True) - np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True))

    # epochs
    num_epochs = config["num_epochs"]
    epoch_eval = config["epoch_eval"]

    #split data
    len_train = int(0.8*len(sdfs))
    
    # Set the parameters
    lower_bound = 0
    upper_bound = len(sdfs)-1
    # Generate random integers without replacement
    selected_integers = np.random.choice(np.arange(lower_bound, upper_bound + 1), len_train, replace=False)
    # Calculate the list of all other integers
    all_integers = set(np.arange(lower_bound, upper_bound + 1))
    other_integers = all_integers.difference(selected_integers)
    other_integers_list = list(other_integers)
    train_data = sdfs[selected_integers]
    val_data = sdfs[other_integers_list]

    # datasets
    train_dataset = PoseuilleFlowSDF(train_data)
    val_dataset = PoseuilleFlowSDF(val_data)

    # dataloaders
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # setup models
    model = ConvNet(
        input_size=nvox, 
        channels_in=train_data.shape[1], 
        channels_init=config["channels_init"],
        channels_out=config["channels_out"],
        latent_space_size=config["latent_space_size"],
    )
    model = model.to(device=device)
    model.train()
    print("There are ", sum(p.numel() for p in model.parameters()), " parameters to train.")

    # Creates a GradScaler once at the beginning of training.
    # scaler = GradScaler()
    
    # set the parameters and the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # if load previous model
    if "continued" in model_name:
        continued_from = model_name.split("continued")[0][:-1]
        checkpoint = torch.load(f'geometry_trainings/geometry_saved_models/{continued_from}')
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    
    # set up scheduler
    if config["scheduler"]:
        scheduler_patience = config["scheduler_patience"]
        if config["scheduler_type"] == "exponential":
            scheduler = ExponentialLR(optimizer, config["scheduler_gamma"])
        elif config["scheduler_type"] == "cyclic":
            scheduler_base_lr = config["scheduler_base_lr"]
            scheduler_max_lr = config["scheduler_max_lr"]
            scheduler = CyclicLR(optimizer, base_lr=scheduler_base_lr, max_lr=scheduler_max_lr, cycle_momentum=False)

    # set up losses
    loss_function = config["loss_function"]
    if loss_function == "mae":
        loss_function = L1Loss()
    else:
        loss_function = MSELoss()

    # setup the writer
    writer = SummaryWriter(log_dir="geometry_trainings/geometry_logs/" + model_name)

    print("Setup the model, dataloader, datasets, loss funcitons, optimizers.")     
    
    for epoch in tqdm(range(num_epochs)):
        # initialize iterations loss to 0
        epoch_total_loss = 0

        # iterate through the dataset
        for sdf_batch in tqdm(train_dataloader):
            sdf_batch = sdf_batch.to(device=device, dtype=dtype) # move to device, e.g. GPU

            # make sure there is no gradient
            optimizer.zero_grad()

            # =====================forward======================
            # Runs the forward pass with autocasting.
            # with autocast(device_type='cuda', dtype=torch.float16):
            # compute latent vectors
            reconstruction = model(sdf_batch)
            loss = loss_function(reconstruction, sdf_batch)

            # add sparsity loss for the first 100 epochs to prevent all 0s
            if epoch < epoch_sparsity and not("continued" in model_name):
                sparsity_weight = 1
                loss += sparsity_weight * torch.mean(torch.abs(reconstruction.mean(dim=(0)) - target_sparsity))

            # ===================backward====================
            loss.backward()
            optimizer.step()
            
            # # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            # scaler.scale(loss).backward()
            # # otherwise, optimizer.step() is skipped.
            # scaler.step(optimizer)
            # # Updates the scale for next iteration.
            # scaler.update()

            # update write iteration loss
            epoch_total_loss += (sdf_batch.shape[0]*loss.item())
            
        # write to tensorboard
        writer.add_scalar("Loss/train", epoch_total_loss/len_train, epoch)
        print(
            f"Total: {epoch_total_loss/len_train:.3e}, ",
        )
        
        # scheduler and eval every n steps
        if config["scheduler"]:
            if not((epoch+1)%scheduler_patience): 
                scheduler.step()
                print("Current learning rate: ", scheduler.get_last_lr())
        
        if not((epoch+1)%epoch_eval):
            with torch.no_grad():
                # evaluate the model
                model.eval()
                epoch_total_val_loss = 0

                # iterate through the dataset
                for sdf_batch in tqdm(val_dataloader):
                    sdf_batch = sdf_batch.to(device=device, dtype=dtype) # move to device, e.g. GPU
                    reconstruction = model(sdf_batch)
                    val_loss = loss_function(reconstruction, sdf_batch)
                    # update write iteration loss
                    epoch_total_val_loss += (sdf_batch.shape[0]*val_loss.item())

                # write it on the same plot
                writer.add_scalar("Loss/val", epoch_total_val_loss/(val_dataset.__len__()), epoch)
                print(
                    f"Val total: {epoch_total_val_loss/(val_dataset.__len__()):.3e}, ",
                )
            # set it back to train mode
            model.train()

        # save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            }, 
            "geometry_trainings/geometry_saved_models/" + model_name
        )
        print(f"Saved model, epoch {epoch}.")
