# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import yaml
from data.data_generation import generate_dataset
from helper_functions_sampling import get_all_points, PointPooling3D
from train_script import PoseuilleFlowAnalytic
from torch.utils.data import DataLoader
import torch
from networks_models import ConvUNetBis, SmallLinear
import numpy as np
from tqdm import tqdm
from loss_functions import PDELoss, ReconLoss
import pdb



if __name__ == "__main__":
    print("hi")
    # loading the model config
    with open("training_config.yml") as file:
        config = yaml.safe_load(file)

    # setting device and data types
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # boundary loss weights parameters
    boundary_recon = config["boundary_recon"]
    boundary_pde = 1 - boundary_recon
    # flow loss weights parameters
    flow_pde = config["flow_pde"]
    flow_recon = 1 - flow_pde
    # background loss weights parameters
    background_recon = config["background_recon"]
    background_pde = 1 - background_recon

    # set up losses
    recon_loss_function = ReconLoss()
    pde_loss_function = PDELoss(rho=config["rho"], mu=config["mu"], gx=config["gx"], gy=config["gy"], gz=config["gz"])

    # volume size
    nvox = config["nvox"]
    nsamples = 2 #config["nsamples"]
    samples, segmentation_maps = generate_dataset(nsamples=nsamples, nvox=nvox)
    val_dataset = PoseuilleFlowAnalytic(samples, segmentation_maps)
    batch_size = config["batch_size"]
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # setup models
    model = ConvUNetBis(input_size=64, channels_in=samples.shape[1], channels_init=4)
    model = model.to(device=device)
    model.eval()
    print("There are ", sum(p.numel() for p in model.parameters()), " parameters to train.")

    smallLinear = SmallLinear(num_features=model.channels_out+4, num_outputs=4) # + x, y, segmentation label
    smallLinear.eval()
    print("There are ", sum(p.numel() for p in smallLinear.parameters()), " parameters to train.")

    model_name = config["model_name"]
    checkpoint = torch.load(f'trainings/saved_models/{model_name}')
    model.load_state_dict(checkpoint["model_state_dict"])
    smallLinear.load_state_dict(checkpoint["linear_model_state_dict"])

    # point sampler
    pp3d = PointPooling3D(interpolation="trilinear")

    epoch_total_loss = 0
    epoch_recon_loss = 0
    epoch_pde_loss = 0

    # predictions = torch.from_numpy(np.zeros_like(samples))
    # predictions = predictions.float()

    # iterate through the dataset
    for i, (flow_sample, map_sample) in enumerate(val_dataloader):
        flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU
        map_sample = map_sample.to(device=device, dtype=dtype)

        # =====================forward======================
        # compute latent vectors
        latent_vectors = model(flow_sample)

        # select same number of points per image to sample, unsqueeze at dim 1 to get the shape
        # batch_size x 1 x num_points x coordinates_size
        pts_ints = torch.Tensor(np.array([get_all_points(map_one.squeeze()) for map_one in map_sample])).unsqueeze(1).unsqueeze(1)
        # normalize the points to be in -1, 1 (required by grid_sample)
        pts = 2*(pts_ints/(nvox-1)) - 1

        # features, segmentation maps, flow values
        pts_vectors, pts_maps, pts_flows = pp3d(latent_vectors, map_sample, flow_sample, pts)
        pts_locations = pts.squeeze()
        pts_locations.requires_grad = True # needed for the PDE loss

        # create feature vectors for each sampled point
        feature_vector = torch.cat([pts_locations, pts_maps, pts_vectors], dim=-1) 
        feature_vector = feature_vector.reshape((-1, model.channels_out + 4)) # x, y, z, seg inter, features

        # split input features to allow taking separate derivatives
        inputs = [feature_vector[..., i:i+1] for i in range(feature_vector.shape[-1])]
        x_ = torch.cat(inputs, axis=-1)
        
        # forward through linear model
        outputs_linear = smallLinear(x_)
        # print(predictions[i*batch_size:batch_size*(i+1)].shape)
        print(pts_ints.squeeze().shape)
        print(outputs_linear.shape)
        print(predictions[i*batch_size].shape)
        # predictions[i*batch_size:batch_size*(i+1)][0][pts_ints.squeeze().long()] = outputs_linear[0, ..., :-1]
        # predictions[i*batch_size:batch_size*(i+1)][pts_ints.squeeze().long()] = outputs_linear[0, ..., :-1]

        # =====================losses======================
        # get the losses weights for each point
        num_loss_terms = 2
        seg_interpolations_rand = pts_maps.reshape((-1, 1))
        weights = torch.ones((len(seg_interpolations_rand), num_loss_terms), dtype=torch.float)
        # boundary weights
        weights[(seg_interpolations_rand>=0.75).squeeze(), :] = torch.Tensor([boundary_pde, boundary_recon])
        # flow weights
        weights[(seg_interpolations_rand>=1.25).squeeze(), :] = torch.Tensor([flow_pde, flow_recon])
        # background weights
        weights[(seg_interpolations_rand<0.5).squeeze(), :] = torch.Tensor([background_pde, background_recon])

        # compute the loss
        pde_loss = weights[:, 0]*pde_loss_function.compute_loss(inputs, outputs_linear)
        recon_loss = weights[:, 1]*recon_loss_function.compute_loss(pts_flows, outputs_linear)
        loss = torch.mean(pde_loss + recon_loss)

        # update write iteration loss
        epoch_total_loss += loss.item()
        epoch_recon_loss += recon_loss.mean().item()
        epoch_pde_loss += pde_loss.mean().item()

    print("Total loss: ", epoch_total_loss, "Reconstruction loss: ", epoch_recon_loss, "PDE loss: ", epoch_pde_loss)
    # np.save("predictions.npy", predictions.detach().cpu().numpy())
