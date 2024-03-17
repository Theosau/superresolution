import torch, pdb, os, yaml, shutil, importlib, math
import numpy as np
from tqdm import tqdm
from pinn_models import PinnNet, PinnNetConcat, SmallPinnNet, LargePinnNet, ExtraLargePinnNet
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import GradScaler 
from torch import autocast

import sys
sys.path.append('/notebooks/superresolution/')
curdir = os.getcwd()
os.chdir('..')
from data.data_generation import generate_dataset, generate_pipe_flow_dataset, generate_3dstenosis, generate_3daneurysm
from helper_functions_sampling import get_points, PointPooling3D
from loss_functions import PDELoss, ReconLoss, ExactReconLoss, VelocityNormLoss
os.chdir(curdir)

# Custom dataset
class PoseuilleFlowAnalytic(Dataset):
    def __init__(self, flow, seg_map, velocity_scale, sdf):
        super(PoseuilleFlowAnalytic, self).__init__()
        self.flow = flow
        self.seg_map = seg_map
        self.vel_scale = velocity_scale
        self.sdf = sdf
        return
    
    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        return self.flow[idx], self.seg_map[idx], self.vel_scale[idx], self.sdf[idx]

def get_points_vector(
        nvox,
        device,
        map_sample,
        flow_sample,
        vel_scale_sample,
        gf_latent,
        nbackground=0, 
        nboundary=10, 
        nflow=100,
    ):
    # select same number of points per image to sample, unsqueeze at dim 1 to get the shape
    pts = torch.cat(
        [
            get_points(map_one.squeeze(), device, nbackground=nbackground, nboundary=nboundary, nflow=nflow).unsqueeze(0) for map_one in map_sample # map_sample
        ], 
    dim=0).unsqueeze(1).unsqueeze(1)

    # normalize the points to be in [-1, 1] for grid_sample
    pts = 2*(pts/(nvox-1)) - 1

    # move points away from the exact input voxel locations and load them
    pts_rand = torch.clip(pts + (torch.rand_like(pts)*2-1)/(2*nvox), min=-1, max=1) # move to voxels boundaries

    # features, segmentation maps, flow values
    pts_maps_rand, pts_flows_rand = pp3d(map_sample, flow_sample, pts_rand)

    # # update the points to nondimensional scale [0, 1]
    pts_rand = (pts_rand + 1)/2

    # require grad for PDE loss
    pts_locations_rand = pts_rand.squeeze()
    pts_locations_rand.requires_grad = True

    # flow and geometry feautres repeated
    gf_repeated = torch.cat([a.repeat((1, nbackground + nboundary + nflow, 1)) for a in gf_latent], dim=0)

    # create feature vectors for each sampled point
    feature_vector = torch.cat([pts_locations_rand, gf_repeated], dim=-1)
    feature_vector = feature_vector.reshape((-1, feature_vector.shape[-1])) #  x, y, z, features

    # split input features to allow taking separate derivatives
    inputs = [feature_vector[..., i:i+1] for i in range(feature_vector.shape[-1])]
    x_ = torch.cat(inputs, axis=-1)
    
    # get tensor for velocity scale (same for all actually)
    vel_scale_repeated = torch.cat([a.repeat(nbackground + nboundary + nflow) for a in vel_scale_sample])

    return x_, inputs, pts_flows_rand, pts_maps_rand, vel_scale_repeated


def get_losses_weights(
        pts_maps_rand, 
        device, 
        dtype,
        boundary_pde, 
        boundary_recon, 
        boundary_exact,
        flow_pde, 
        flow_recon, 
        flow_exact,
        background_pde, 
        background_recon, 
        background_exact
    ):
    num_loss_terms = 3
    seg_interpolations_rand = pts_maps_rand.reshape((-1, 1))

    # split the three types of points
    boundary_points = torch.logical_and(
        seg_interpolations_rand >= 0.1,
        seg_interpolations_rand<1.5
    ).squeeze()
    flow_points = (seg_interpolations_rand>=1.5).squeeze()
    # background_points = (seg_interpolations_rand<0.1).squeeze()

    # set the weights of each function for the type of points
    weights = torch.ones((len(seg_interpolations_rand), num_loss_terms), dtype=torch.float, device=device)
    # boundary weights
    weights[boundary_points, :] = torch.tensor(
        [boundary_pde, boundary_recon, boundary_exact],
        device=device,
        dtype=dtype
    )
    # flow recon weights
    weights[flow_points, :] = torch.tensor(
        [flow_pde, flow_recon, flow_exact],
        device=device,
        dtype=dtype
    )
    return weights, flow_points, boundary_points



if __name__ == "__main__":

    # loading the model config
    with open("pinn_training_config.yml") as file:
        config = yaml.safe_load(file)

    # model name
    model_name = config["model_name"]

    # save config file for reload
    shutil.copyfile("pinn_training_config.yml", f"pinn_trainings/pinn_config_models/{model_name}.yml")
    shutil.copyfile("pinn_models.py", f"pinn_trainings/pinn_models/{model_name}.py")

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
    
    # datasets
    gen_stenosis = config["gen_stenosis"]
    gen_aneurysm = config["gen_aneurysm"]
    assert gen_stenosis or gen_aneurysm, 'You should at least generate stenosis flows or aneurysm flows.'
    
    if gen_stenosis:
        flows, segmentation_maps, velocity_scales, sdfs, velocity_scale_for_norm_stenosis = generate_3dstenosis(
            nvox=64, 
            sdf=True,
            transforms=transforms,
            data_aug=data_aug
        )
    
    if gen_aneurysm:
        flows_aneurysm, segmentation_maps_aneurysm, velocity_scales_aneurysm, sdfs_aneurysm, velocity_scale_for_norm_aneurysm = generate_3daneurysm(
            nvox=64, 
            sdf=True,
            transforms=transforms,
            data_aug=data_aug
        )
    
    if gen_stenosis and gen_aneurysm:
        
        # concatenate vel_scales, maps
        segmentation_maps = np.concatenate((segmentation_maps, segmentation_maps_aneurysm), axis=0)
        del segmentation_maps_aneurysm
        velocity_scales = np.concatenate((velocity_scales, velocity_scales_aneurysm), axis=0)
        del velocity_scales_aneurysm

        # normalize all the flows together
        if velocity_scale_for_norm_stenosis >= velocity_scale_for_norm_aneurysm:
            flows_aneurysm = flows_aneurysm * velocity_scale_for_norm_aneurysm / velocity_scale_for_norm_stenosis
            velocity_scale_for_norm = velocity_scale_for_norm_stenosis
        else:
            flows = flows * velocity_scale_for_norm_stenosis / velocity_scale_for_norm_aneurysm
            velocity_scale_for_norm = velocity_scale_for_norm_aneurysm
        
        # concatenate flows, maps
        flows = np.concatenate((flows, flows_aneurysm), axis=0)
        del flows_aneurysm
    
        # concatenate sdfs all together
        sdfs = np.concatenate((sdfs, sdfs_aneurysm), axis=0)
        del sdfs_aneurysm
    
    # just channel flows
    elif gen_stenosis:
        velocity_scale_for_norm = velocity_scale_for_norm_stenosis

    # just aneurysm flows
    else:
        flows = flows_paneurysm
        segmentation_maps = segmentation_maps_aneurysm 
        velocity_scales = velocity_scales_aneurysm 
        sdfs = sdfs_aneurysm 
        velocity_scale_for_norm = velocity_scale_for_norm_aneurysm 
    
    sdfs_min = np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True)
    sdfs_max = np.max(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True)
    # normalize the signed distance fields
    sdfs = (sdfs - sdfs_min)/(sdfs_max - sdfs_min)
    
    
    if config["noisy"]:
        print('Adding noise.')
        # Create the Gaussian noise
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
        del noise
        del boundaries
        del no_slip
        del segmentation_maps_rep
        print('Deleted arrays.')

    # boundary loss weights parameters
    boundary_recon = config["boundary_recon"]
    boundary_pde = config["boundary_pde"]
    boundary_exact = config["boundary_exact"]
    # flow loss weights parameters
    flow_pde = config["flow_pde"]
    flow_recon = config["flow_recon"]
    flow_exact = config["flow_exact"]
    # background loss weights parameters
    background_recon = config["background_recon"]
    background_pde = config["background_pde"]
    background_exact = config["background_exact"]

    # epochs
    num_epochs = config["num_epochs"]

    #split data
    len_train = math.floor(0.8*len(flows))

    # Set the parameters
    lower_bound = 0
    upper_bound = len(flows)-1
    # Generate random integers without replacement
    selected_integers = np.random.choice(np.arange(lower_bound, upper_bound + 1), len_train, replace=False)
    # Calculate the list of all other integers
    all_integers = set(np.arange(lower_bound, upper_bound + 1))
    other_integers_list = list(all_integers.difference(selected_integers))
    train_data = flows[selected_integers]
    val_data = flows[other_integers_list]

    # define the training samples
    train_data = flows[selected_integers]
    train_maps = segmentation_maps[selected_integers]
    train_scales = velocity_scale_for_norm.squeeze()*np.ones_like(velocity_scales[selected_integers])
    train_sdfs = sdfs[selected_integers]

    # define the validation samples
    val_data = flows[other_integers_list]
    val_maps = segmentation_maps[other_integers_list]
    val_scales = velocity_scale_for_norm.squeeze()*np.ones_like(velocity_scales[other_integers_list])
    val_sdfs = sdfs[other_integers_list]

    # delete the lists
    del all_integers
    # del other_integers_list
    
    # build datasets
    train_dataset = PoseuilleFlowAnalytic(train_data, train_maps, train_scales, train_sdfs)
    val_dataset = PoseuilleFlowAnalytic(val_data, val_maps, val_scales, val_sdfs)

    # build dataloaders
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # setup models

    # geometry model
    geometry_model_name = config['geometry_model_name']
    geometry_specific_model = importlib.import_module(f'geometry_submodule.geometry_trainings.geometry_models.{geometry_model_name}')
    ConvNetGeo = geometry_specific_model.ConvNet
    os.chdir('../geometry_submodule')
    with open(f"geometry_trainings/geometry_config_models/{geometry_model_name}.yml") as file:
        geometry_config = yaml.safe_load(file)
    checkpoint = torch.load(f"geometry_trainings/geometry_saved_models/{geometry_model_name}")
    os.chdir(curdir)
    geometry_model = ConvNetGeo(
        input_size=nvox, 
        channels_in=sdfs.shape[1], 
        channels_init=geometry_config["channels_init"],
        channels_out=geometry_config["channels_out"],
        latent_space_size=geometry_config["latent_space_size"],
    )
    geometry_model.load_state_dict(checkpoint["model_state_dict"])
    geometry_model = geometry_model.to(device=device)
    geometry_model.eval()

    # flow model
    flow_model_name = config['flow_model_name']
    flow_specific_model = importlib.import_module(f'flow_submodule.flow_trainings.flow_models.{flow_model_name}')
    os.chdir('../flow_submodule')
    ConvNetFlow = flow_specific_model.ConvNet
    with open(f"flow_trainings/flow_config_models/{flow_model_name}.yml") as file:
        flow_config = yaml.safe_load(file)
    checkpoint = torch.load(f"flow_trainings/flow_saved_models/{flow_model_name}")
    os.chdir(curdir)
    flow_model = ConvNetFlow(
        input_size=nvox, 
        channels_in=flows.shape[1], 
        channels_init=flow_config["channels_init"],
        channels_out=flow_config["channels_out"],
        latent_space_size=flow_config["latent_space_size"],
    )
    flow_model.load_state_dict(checkpoint["model_state_dict"])
    flow_model = flow_model.to(device=device)
    flow_model.eval()

    # to get the latent outputs
    intermediate_outputs = []
    def hook(module, input, output):
        intermediate_outputs.append(output)
    geometry_model.enc7.register_forward_hook(hook)
    flow_model.enc7.register_forward_hook(hook)

    # pinn  model
    if config["model_type"] == "pinn":
        model = PinnNet(
            num_features=3+geometry_config["latent_space_size"]+flow_config["latent_space_size"], 
            num_outputs=4,
        )
    elif config["model_type"]=="pinn_small":
        model = SmallPinnNet(
            num_features=3+geometry_config["latent_space_size"]+flow_config["latent_space_size"], 
            num_outputs=4,
        )
    elif config["model_type"]=="pinn_large":
        model = LargePinnNet(
            num_features=3+geometry_config["latent_space_size"]+flow_config["latent_space_size"], 
            num_outputs=4,
        )
    elif config["model_type"]=="pinn_extra_large":
        model = ExtraLargePinnNet(
            num_features=3+geometry_config["latent_space_size"]+flow_config["latent_space_size"], 
            num_outputs=4,
        )
    elif config["model_type"] == "pinn_concat":
        model = PinnNetConcat(
            num_features=3+geometry_config["latent_space_size"]+flow_config["latent_space_size"], 
            num_outputs=4,
            num_con_input=3+geometry_config["latent_space_size"]+flow_config["latent_space_size"]
        )
    model = model.to(device=device)
    model.train()
    print("There are ", sum(p.numel() for p in model.parameters()), " parameters to train.")

    # set the parameters and the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # if load previous model
    if "continued" in model_name:
        continued_from = model_name.split("continued")[0][:-1]
        print(f"Loading {continued_from}")
        checkpoint = torch.load(f'pinn_trainings/pinn_saved_models/{continued_from}')
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

    # set up scheduler
    if config["scheduler"]:
        scheduler = ExponentialLR(optimizer, config["scheduler_gamma"])
        scheduler_patience = config["scheduler_patience"]

    # set up losses
    recon_loss_function = ReconLoss()
    exact_recon_loss_function = ExactReconLoss()
    pde_loss_function = PDELoss(
        rho=config["rho"],
        mu=config["mu"],
        gx=config["gx"],
        gy=config["gy"],
        gz=config["gz"],
        space_scale=20, # 10 for basic, now 20! added for new space scale
    )
    vel_norm_function = VelocityNormLoss()
    pde_multiplier = config["pde_multiplier"]

    # setup point sampler
    pp3d = PointPooling3D(interpolation="trilinear")

    # setup the writer
    writer = SummaryWriter(log_dir="pinn_trainings/pinn_logs/" + model_name)
    # Creates a GradScaler once at the beginning of training for mixed precision
    # scaler = GradScaler()

    print("Setup the model, dataloader, datasets, loss funcitons, optimizers.")
    
    for epoch in tqdm(range(num_epochs)):
        # initialize iterations loss to 0
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_pde_loss = 0
        epoch_continuity_residual = 0
        epoch_nse_x_residual = 0
        epoch_nse_y_residual = 0
        epoch_nse_z_residual = 0
        # epoch_pressure_residual = 0
        epoch_dpdy = 0
        epoch_dpdx = 0
        epoch_dpdz = 0

        # iterate through the dataset
        for i, (flow_sample, map_sample, vel_scale_sample, sdf_sample) in enumerate(train_dataloader):
            flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU
            map_sample = map_sample.to(device=device, dtype=dtype)
            vel_scale_sample = vel_scale_sample.to(device=device, dtype=dtype)
            sdf_sample = sdf_sample.to(device=device, dtype=dtype)

            intermediate_outputs = []
            with torch.no_grad():
                _ = geometry_model(sdf_sample)
                _ = flow_model(flow_sample)
                geometry_latent = intermediate_outputs[0].clone().detach().squeeze(-1).squeeze(-1).squeeze(-1)
                flow_latent = intermediate_outputs[1].clone().detach().squeeze(-1).squeeze(-1).squeeze(-1)
                gf_latent = torch.cat([geometry_latent, flow_latent], dim=-1)
            
            # make sure there is no gradient
            optimizer.zero_grad()
            
            # =====================forward======================
            
            # generate the feature vectors for each point
            x_, inputs, pts_flows_rand, pts_maps_rand, vel_scale_repeated = get_points_vector(
                nvox,
                device,
                map_sample,
                flow_sample,
                vel_scale_sample,
                gf_latent,
                nbackground=0, 
                nboundary=10, #20
                nflow=20,
            )
            
            #### Adding mixed precision training for faster iterations
            # Runs the forward pass with autocasting.
            # with autocast(device_type='cuda', dtype=torch.float16):
            # forward through linear model
            outputs_linear = model(x_)

            # =====================losses======================
            # get the losses weights for each point
            weights, flow_points, boundary_points = get_losses_weights(
                pts_maps_rand, 
                device, 
                dtype,
                boundary_pde, 
                boundary_recon, 
                boundary_exact,
                flow_pde, 
                flow_recon, 
                flow_exact,
                background_pde, 
                background_recon, 
                background_exact
            )


            ## compute the losses
            # pde loss
            use_pde = True if torch.sum(weights[:, 0]==1.0)>0 else False
            if use_pde:
                # gets the squared reisudal pointwise
                continuity_residual, nse_x_residual, nse_y_residual, nse_z_residual, dpdy, dpdx, dpdz = pde_loss_function.compute_loss(inputs, outputs_linear, vel_scale_repeated)

                # take the mean of the losses
                continuity_residual = torch.sum(weights[:, 0]*continuity_residual)/torch.sum(weights[:, 0]==1.0) 
                nse_x_residual = torch.sum(weights[:, 0]*nse_x_residual)/torch.sum(weights[:, 0]==1.0)
                nse_y_residual = torch.sum(weights[:, 0]*nse_y_residual)/torch.sum(weights[:, 0]==1.0) 
                nse_z_residual = torch.sum(weights[:, 0]*nse_z_residual)/torch.sum(weights[:, 0]==1.0)

                # get the pressure gradients for tensorboard
                dpdy = torch.sum(weights[:, 0]*(torch.abs(dpdy)))/torch.sum(weights[:, 0]==1.0)
                dpdx = torch.sum(weights[:, 0]*(torch.abs(dpdx)))/torch.sum(weights[:, 0]==1.0)
                dpdz = torch.sum(weights[:, 0]*(torch.abs(dpdz)))/torch.sum(weights[:, 0]==1.0)

                # total pde loss
                pde_loss = continuity_residual + nse_x_residual + nse_y_residual + nse_z_residual

            else:
                # set to 0 if no point uses the PDE
                pde_loss = torch.tensor(0, dtype=dtype, device=device) 

            # reconstruction loss
            use_recon = True
            if use_recon:
                recon_loss = weights[:, 1]*recon_loss_function.compute_loss(pts_flows_rand, outputs_linear, epoch)
                # compute the mean of the loss
                recon_loss = torch.sum(recon_loss)/torch.sum(weights[:, 1]==1.0)
            else:
                recon_loss = torch.tensor(0, dtype=dtype, device=device)   

            # total losses
            loss = recon_loss + pde_multiplier*pde_loss

            # ===================backward====================
            loss.backward()
            optimizer.step()
            
            # # ===================backward with mixed precision====================
            # # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # # Backward passes under autocast are not recommended.
            # # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            # scaler.scale(loss).backward()
            # # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # # otherwise, optimizer.step() is skipped.
            # scaler.step(optimizer)
            # # Updates the scale for next iteration.
            # scaler.update()

            # update write iteration loss
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item() 
            epoch_pde_loss += pde_loss.item()
            # specific pde
            if use_pde:
                epoch_continuity_residual += continuity_residual.item()
                epoch_nse_x_residual += nse_x_residual.item()
                epoch_nse_y_residual += nse_y_residual.item()
                epoch_nse_z_residual += nse_z_residual.item()
                # epoch_pressure_residual += pressure_residual.item()
                epoch_dpdy += dpdy.item()
                epoch_dpdx += dpdx.item()
                epoch_dpdz += dpdz.item()

        # write to tensorboard
        writer.add_scalar("Loss/train", epoch_total_loss/(i+1), epoch)
        writer.add_scalar("Reconstruction Loss/train", epoch_recon_loss/(i+1), epoch)
        writer.add_scalar("PDE Loss/train", epoch_pde_loss/(i+1), epoch)

        # pde terms
        if use_pde:
            writer.add_scalar("PDE Loss/Continuity", epoch_continuity_residual/(i+1), epoch)
            writer.add_scalar("PDE Loss/xmtm", epoch_nse_x_residual/(i+1), epoch)
            writer.add_scalar("PDE Loss/ymtm", epoch_nse_y_residual/(i+1), epoch)
            writer.add_scalar("PDE Loss/zmtm", epoch_nse_z_residual/(i+1), epoch)
            # writer.add_scalar("PDE Loss/Pressure", epoch_pressure_residual, epoch)
            writer.add_scalar("Dpdy", epoch_dpdy/(i+1), epoch)
            writer.add_scalar("Dpdx", epoch_dpdx/(i+1), epoch)
            writer.add_scalar("Dpdz", epoch_dpdz/(i+1), epoch)

            # points
            writer.add_scalar("Points/flowPDE", flow_points.sum(), epoch)
            writer.add_scalar("Points/boundary", boundary_points.sum(), epoch)
            # writer.add_scalar("Points/background", background_points.sum(), epoch)
        
        print(
                f"Total: {epoch_total_loss/(i+1):.3e}, ", 
                f"Reconstruction: {epoch_recon_loss/(i+1):.3e}, ", 
                f"PDE: {epoch_pde_loss/(i+1):.3e}, ",
            )
        
        #################################### EVALUATION ##########################################
        # at the end of each epoch
        if not((epoch+1)%config["scheduler_patience"]):
            if config["scheduler"]:
             # scheduler every n steps
                scheduler.step()

            # evaluate the validation dataset
            # initialize iterations loss to 0
            epoch_total_loss_val = 0
            epoch_recon_loss_val = 0
            epoch_pde_loss_val = 0
            
            # iterate through the dataset
            for i, (flow_sample, map_sample, vel_scale_sample, sdf_sample) in enumerate(val_dataloader):
                flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU
                map_sample = map_sample.to(device=device, dtype=dtype)
                vel_scale_sample = vel_scale_sample.to(device=device, dtype=dtype)
                sdf_sample = sdf_sample.to(device=device, dtype=dtype)

                intermediate_outputs = []
                with torch.no_grad():
                    _ = geometry_model(sdf_sample)
                    _ = flow_model(flow_sample)
                    geometry_latent = intermediate_outputs[0].clone().detach().squeeze(-1).squeeze(-1).squeeze(-1)
                    flow_latent = intermediate_outputs[1].clone().detach().squeeze(-1).squeeze(-1).squeeze(-1)
                    gf_latent = torch.cat([geometry_latent, flow_latent], dim=-1)

                # make sure there is no gradient
                optimizer.zero_grad()

                # =====================forward======================

                # generate the feature vectors for each point
                x_, inputs, pts_flows_rand, pts_maps_rand, vel_scale_repeated = get_points_vector(
                    nvox,
                    device,
                    map_sample,
                    flow_sample,
                    vel_scale_sample,
                    gf_latent,
                    nbackground=0, 
                    nboundary=10, #20 
                    nflow=20,
                )

                # forward through linear model
                outputs_linear = model(x_)

                # =====================losses======================
                # get the losses weights for each point
                weights, flow_points, boundary_points = get_losses_weights(
                    pts_maps_rand, 
                    device, 
                    dtype,
                    boundary_pde, 
                    boundary_recon, 
                    boundary_exact,
                    flow_pde, 
                    flow_recon, 
                    flow_exact,
                    background_pde, 
                    background_recon, 
                    background_exact
                )

                ## compute the losses
                # pde loss
                use_pde = True if torch.sum(weights[:, 0]==1.0)>0 else False
                if use_pde:
                    # gets the squared reisudal pointwise
                    continuity_residual, nse_x_residual, nse_y_residual, nse_z_residual, dpdy, dpdx, dpdz = pde_loss_function.compute_loss(inputs, outputs_linear, vel_scale_repeated)

                    # take the mean of the losses
                    continuity_residual = torch.sum(weights[:, 0]*continuity_residual)/torch.sum(weights[:, 0]==1.0) 
                    nse_x_residual = torch.sum(weights[:, 0]*nse_x_residual)/torch.sum(weights[:, 0]==1.0)
                    nse_y_residual = torch.sum(weights[:, 0]*nse_y_residual)/torch.sum(weights[:, 0]==1.0) 
                    nse_z_residual = torch.sum(weights[:, 0]*nse_z_residual)/torch.sum(weights[:, 0]==1.0)

                    # total pde loss
                    pde_loss = continuity_residual + nse_x_residual + nse_y_residual + nse_z_residual

                else:
                    # set to 0 if no point uses the PDE
                    pde_loss = torch.tensor(0, dtype=dtype, device=device) 

                # reconstruction loss
                use_recon = True
                if use_recon:
                    recon_loss = weights[:, 1]*recon_loss_function.compute_loss(pts_flows_rand, outputs_linear, epoch)
                    recon_loss = torch.sum(recon_loss)/torch.sum(weights[:, 1]==1.0)
                else:
                    recon_loss = torch.tensor(0, dtype=dtype, device=device)   

                # total losses
                loss = recon_loss + pde_multiplier*pde_loss

                # update write iteration loss
                epoch_total_loss_val += loss.item()
                epoch_recon_loss_val += recon_loss.item() 
                epoch_pde_loss_val += pde_loss.item()

            # write to tensorboard
            writer.add_scalar("Loss/val", epoch_total_loss_val/(i+1), epoch)
            writer.add_scalar("Reconstruction Loss/val", epoch_recon_loss_val/(i+1), epoch)
            writer.add_scalar("PDE Loss/val", epoch_pde_loss_val/(i+1), epoch)

            print(
                f"Total val: {epoch_total_loss_val/(i+1):.3e}, ", 
                f"Reconstruction val: {epoch_recon_loss_val/(i+1):.3e}, ", 
                f"PDE val: {epoch_pde_loss_val/(i+1):.3e}, ",
            )
        #################################### END EVALUATION ##########################################

        # save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "velocity_scale_for_norm":velocity_scale_for_norm,
                "sdfs_min":sdfs_min,
                "sdfs_max":sdfs_max,
                "other_integers_list":other_integers_list,
            }, 
            "pinn_trainings/pinn_saved_models/" + model_name
        )
        print(f"Saved model, epoch {epoch}.")
