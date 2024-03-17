import torch, pdb, os, yaml, shutil
import numpy as np
from tqdm import tqdm
from flow_models import ConvNet
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss, L1Loss, CosineSimilarity
from copy import deepcopy
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
class PoseuilleFlow(Dataset):
    def __init__(self, flow):
        super(PoseuilleFlow, self).__init__()
        self.flow = flow
        return

    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        return self.flow[idx]

if __name__ == "__main__":

    # loading the model config
    with open("flow_training_config.yml") as file:
        config = yaml.safe_load(file)

    # model name
    model_name = config["model_name"]

    # save config file for reload
    shutil.copyfile("flow_training_config.yml", f"flow_trainings/flow_config_models/{model_name}.yml")
    shutil.copyfile("flow_models.py", f"flow_trainings/flow_models/{model_name}.py")

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
        flows, segmentation_maps, _, velocity_scale_for_norm_stenosis = generate_3dstenosis(
            nvox=64, 
            sdf=False, 
            transforms=transforms,
            data_aug=data_aug
        )
        # flows, segmentation_maps, _, velocity_scale_for_norm_channel = generate_dataset(
        #     nsamples=nsamples, 
        #     nvox=nvox, 
        #     xmin=0, 
        #     xmax=10, 
        #     ymin=0, 
        #     ymax=10, 
        #     sdf=False,
        #     flows_in_z = flows_in_z,
        #     multthickness = multthickness,
        #     multpositions = multpositions,
        #     multangles = multangles
        # )
    
    if gen_aneurysm:
        flows_aneurysm, segmentation_maps_aneurysm, _, velocity_scale_for_norm_aneurysm = generate_3daneurysm(
            nvox=64, 
            sdf=False, 
            transforms=transforms,
            data_aug=data_aug
        )
        # flows_pipe, segmentation_maps_pipe, _, velocity_scale_for_norm_pipe = generate_pipe_flow_dataset(
        #     nsamples=nsamples,
        #     nvox=nvox, 
        #     xmin=0, 
        #     xmax=10, 
        #     ymin=0, 
        #     ymax=10, 
        #     sdf=False,
        #     multthickness = multthickness,
        #     multpositions = multpositions,
        #     multangles = multangles
        # )

    if gen_stenosis and gen_aneurysm:
        # concatenate vel_scales, maps
        segmentation_maps = np.concatenate((segmentation_maps, segmentation_maps_aneurysm), axis=0)
        del segmentation_maps_aneurysm

        # normalize all the flows together
        if velocity_scale_for_norm_stenosis >= velocity_scale_for_norm_aneurysm:
            flows_aneurysm = flows_aneurysm * velocity_scale_for_norm_aneurysm / velocity_scale_for_norm_stenosis
            velocity_scale_for_norm = velocity_scale_for_norm_stenosis
        else:
            flows = flows * velocity_scale_for_norm_stenosis / velocity_scale_for_norm_aneursym
            velocity_scale_for_norm = velocity_scale_for_norm_aneurysm
        
        # concatenate flows, maps
        flows = np.concatenate((flows, flows_aneurysm), axis=0)
        del flows_aneurysm

    # just channel flows
    elif gen_stenosis:
        velocity_scale_for_norm = velocity_scale_for_norm_stenosis

    # just pipe flows
    else:
        flows = flows_aneurysm
        segmentation_maps = segmentation_maps_aneurysm
        velocity_scale_for_norm = velocity_scale_for_norm_aneurysm

    # transforms = config["transforms"]
    # data_aug = config["data_aug"]
    # # generate stenosis
    # flows, segmentation_maps, _, velocity_scale_for_norm = generate_3daneurysm(
    #     nvox=64, 
    #     sdf=False, 
    #     transforms=transforms,
    #     data_aug=data_aug
    # )

    if config["noisy"]:      
        print('Adding noise.')
        noise = np.random.normal(loc=0.0, scale=config["noise_std"], size=flows.shape)
        segmentation_maps_rep = np.repeat(segmentation_maps, 3, axis=1)

        # remove noise from background
        noise[segmentation_maps_rep==0]=0

        # remove noise from no slip boundary
        boundaries = segmentation_maps_rep==1
        no_slip = flows==0
        noise[np.logical_and(boundaries, no_slip)] = 0

        # Add the noise to the image
        flows = flows * (1 + noise)
        print('Added noise.')
        
        # delete arrays for memory
        del boundaries
        del no_slip
        del noise
        print('Deleted arrays.')

    # epochs
    num_epochs = config["num_epochs"]
    epoch_eval = config["epoch_eval"]

    #split data
    len_train = int(0.8*len(flows))

    # Set the parameters
    lower_bound = 0
    upper_bound = len(flows)-1
    # Generate random integers without replacement
    selected_integers = np.random.choice(np.arange(lower_bound, upper_bound + 1), len_train, replace=False)
    # Calculate the list of all other integers
    all_integers = set(np.arange(lower_bound, upper_bound + 1))
    other_integers = all_integers.difference(selected_integers)
    other_integers_list = list(other_integers)
    train_data = flows[selected_integers]
    val_data = flows[other_integers_list]
    
    # datasets
    train_dataset = PoseuilleFlow(train_data)
    val_dataset = PoseuilleFlow(val_data)

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
        checkpoint = torch.load(f'flow_trainings/flow_saved_models/{continued_from}')
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
    elif loss_function == "cosine":
        loss_function = CosineSimilarity(dim=1, eps=1e-08)
    else:
        loss_function = MSELoss()

    # define the weighted loss function
    def custom_loss(y_pred, y_true):
        weight = 1.0 / torch.abs(y_true)
        weight_clipped = torch.clamp(weight, min=1, max=1)  # adjust min and max as necessary
        weight_clipped[y_true==0] = 1 # 1 # or else it would focus on background too much
        loss = weight_clipped * torch.square(y_pred - y_true) #**2
        return torch.mean(loss)

    loss_function = custom_loss

    # setup the writer
    writer = SummaryWriter(log_dir="flow_trainings/flow_logs/" + model_name)

    print("Setup the model, dataloader, datasets, loss funcitons, optimizers.")     
    
    for epoch in tqdm(range(num_epochs)):
        # initialize iterations loss to 0
        epoch_total_loss = 0

        # iterate through the dataset
        for flow_batch in tqdm(train_dataloader):
            flow_batch = flow_batch.to(device=device, dtype=dtype) # move to device, e.g. GPU

            # make sure there is no gradient
            optimizer.zero_grad()

            # =====================forward======================
            # Runs the forward pass with autocasting.
            # with autocast(device_type='cuda', dtype=torch.float16):
            # compute latent vectors
            reconstruction = model(flow_batch)
            if config["loss_function"] == "cosine":
                loss = torch.sum(loss_function(reconstruction, flow_batch))/(flow_batch.shape[0]*(nvox**3))
            else:
                loss = loss_function(reconstruction, flow_batch)

            # add sparsity loss for the first 100 epochs to prevent all 0s
            if epoch < epoch_sparsity and not("continued" in model_name):
                sparsity_weight = 1
                target_sparsity = target_sparsity #0.1
                # loss += sparsity_weight * torch.mean(torch.abs(reconstruction.mean(dim=(-1)) - target_sparsity))
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
            
            # ===================storing====================
            # update write iteration loss
            epoch_total_loss += (flow_batch.shape[0]*loss.item())
            
        # write to tensorboard
        writer.add_scalar("Loss/train", epoch_total_loss/len_train, epoch)
        print(
            f"Total: {epoch_total_loss/len_train:.3e}, ",
        )
        
        # scheduler and eval every n steps
        if config["scheduler"]:
            if not((epoch+1)%scheduler_patience): # and scheduler.get_last_lr()[0]>1e-5: 
                scheduler.step()
                print("Current learning rate: ", scheduler.get_last_lr())

        if not((epoch+1)%epoch_eval):
            with torch.no_grad():
                # evaluate the model
                model.eval()
                epoch_total_val_loss = 0

                # iterate through the dataset
                for flow_batch in tqdm(val_dataloader):
                    flow_batch = flow_batch.to(device=device, dtype=dtype) # move to device, e.g. GPU
                    reconstruction = model(flow_batch)

                    if config["loss_function"] == "cosine":
                        val_loss = torch.sum(loss_function(reconstruction, flow_batch))/(flow_batch.shape[0]*(nvox**3))
                    else:
                        val_loss = loss_function(reconstruction, flow_batch)

                    # update write iteration loss
                    epoch_total_val_loss += (flow_batch.shape[0]*val_loss.item())

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
                "velocity_scale_for_norm":velocity_scale_for_norm,
            }, 
            "flow_trainings/flow_saved_models/" + model_name
        )
        print(f"Saved model, epoch {epoch}.")
