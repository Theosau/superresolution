import torch, pdb, os, yaml
import numpy as np
from tqdm import tqdm
from networks_models import ConvUNetBis, SmallLinear
from loss_functions import PDELoss, ReconLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms.functional import normalize
from torch.nn.functional import interpolate
from data.data_generation import generate_dataset
from helper_functions_sampling import get_points, PointPooling3D

# Custom dataset
class PoseuilleFlowAnalytic(Dataset):
    def __init__(self, flow, seg_map):
        super(PoseuilleFlowAnalytic, self).__init__()
        self.flow = flow
        self.seg_map = seg_map
        return
    
    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        return self.flow[idx], self.seg_map[idx]


if __name__ == "__main__":

    # loading the model config
    with open("training_config.yml") as file:
        config = yaml.safe_load(file)

    # setting device and data types
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # volume size
    nvox = config["nvox"]
    nsamples = config["nsamples"]
    samples, segmentation_maps = generate_dataset(nsamples=nsamples, nvox=nvox)

    # boundary loss weights parameters
    boundary_recon = config["boundary_recon"]
    boundary_pde = 1 - boundary_recon
    # flow loss weights parameters
    flow_pde = config["flow_pde"]
    flow_recon = 1 - flow_pde
    # background loss weights parameters
    background_recon = config["background_recon"]
    background_pde = 1 - background_recon

    # epochs
    num_epochs = config["num_epochs"]

    # model name
    model_name = config["model_name"]

    #split data
    len_train = int(0.8*len(samples))
    train_data = samples[:len_train]
    train_maps = segmentation_maps[:len_train]

    val_data = samples[len_train:]
    val_maps = segmentation_maps[len_train:]

    # datasets
    train_dataset = PoseuilleFlowAnalytic(train_data, train_maps)
    val_dataset = PoseuilleFlowAnalytic(val_data, val_maps)

    # dataloaders
    batch_size = config["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # setup models
    model = ConvUNetBis(input_size=64, channels_in=train_data.shape[1], channels_init=4)
    model = model.to(device=device)
    model.train()
    print("There are ", sum(p.numel() for p in model.parameters()), " parameters to train.")

    smallLinear = SmallLinear(num_features=model.channels_out+4, num_outputs=4) # + x, y, segmentation label
    smallLinear.train()
    print("There are ", sum(p.numel() for p in smallLinear.parameters()), " parameters to train.")
    
    # set the parameters and the optimizer
    all_params = list(model.parameters()) + list(smallLinear.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)
    if "continued" in model_name:
        continued_from = model_name.split("continued")[0][:-1]
        checkpoint = torch.load(f'trainings/saved_models/{continued_from}')
        model.load_state_dict(checkpoint["model_state_dict"])
        smallLinear.load_state_dict(checkpoint["linear_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"] 
    scheduler = ExponentialLR(optimizer, config["scheduler_gamma"])
    
    # set up losses
    recon_loss_function = ReconLoss()
    pde_loss_function = PDELoss(rho=config["rho"], mu=config["mu"], gx=config["gx"], gy=config["gy"], gz=config["gz"])

    # setup point sampler
    pp3d = PointPooling3D(interpolation="trilinear")

    # setup the writer
    writer = SummaryWriter(log_dir="trainings/logs/" + model_name)

    print("Setup the model, dataloader, datasets, loss funcitons, optimizers.")     
    for epoch in tqdm(range(num_epochs)):

        # initialize iterations loss to 0
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_pde_loss = 0

        # iterate through the dataset
        for flow_sample, map_sample in tqdm(train_dataloader):
            flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU
            map_sample = map_sample.to(device=device, dtype=dtype)

            # make sure there is no gradient
            optimizer.zero_grad()

            # =====================forward======================
            # compute latent vectors
            latent_vectors = model(flow_sample)

            # select same number of points per image to sample, unsqueeze at dim 1 to get the shape
            # batch_size x 1 x num_points x coordinates_size
            pts = torch.Tensor(np.array([get_points(map_one.squeeze()) for map_one in map_sample])).unsqueeze(1).unsqueeze(1)
            # normalize the points to be in -1, 1 (required by grid_sample)
            pts = 2*(pts/(nvox-1)) - 1
            pts.shape

            # move points away from the exact input voxel locations and load them
            pts_rand = torch.clip(pts + (torch.rand_like(pts)*2-1)/(2*nvox), min=-1, max=1) # play around to move more or less around voxels
            # features, segmentation maps, flow values
            pts_vectors_rand, pts_maps_rand, pts_flows_rand = pp3d(latent_vectors, map_sample, flow_sample, pts_rand)
            pts_locations_rand = pts_rand.squeeze()
            pts_locations_rand.requires_grad = True # needed for the PDE loss

            # create feature vectors for each sampled point
            feature_vector = torch.cat([pts_locations_rand, pts_maps_rand, pts_vectors_rand], dim=-1) 
            feature_vector = feature_vector.reshape((-1, model.channels_out + 4)) # x, y, z, seg inter, features

            # split input features to allow taking separate derivatives
            inputs = [feature_vector[..., i:i+1] for i in range(feature_vector.shape[-1])]
            x_ = torch.cat(inputs, axis=-1)
            
            # forward through linear model
            outputs_linear = smallLinear(x_)


            # =====================losses======================
            # get the losses weights for each point
            num_loss_terms = 2
            seg_interpolations_rand = pts_maps_rand.reshape((-1, 1))
            weights = torch.ones((len(seg_interpolations_rand), num_loss_terms), dtype=torch.float)
            # boundary weights
            weights[(seg_interpolations_rand>=0.75).squeeze(), :] = torch.Tensor([boundary_pde, boundary_recon])
            # flow weights
            weights[(seg_interpolations_rand>=1.25).squeeze(), :] = torch.Tensor([flow_pde, flow_recon])
            # background weights
            weights[(seg_interpolations_rand<0.5).squeeze(), :] = torch.Tensor([background_pde, background_recon])

            # compute the loss
            pde_loss = weights[:, 0]*pde_loss_function.compute_loss(inputs, outputs_linear)
            recon_loss = weights[:, 1]*recon_loss_function.compute_loss(pts_flows_rand, outputs_linear)
            loss = torch.mean(pde_loss + recon_loss)

            # ===================backward====================
            loss.backward()
            optimizer.step()

            # update write iteration loss
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.mean().item()
            epoch_pde_loss += pde_loss.mean().item()

        # at the end of each epoch
        scheduler.step()

        # write to tensorboard
        writer.add_scalar("Loss/train", epoch_total_loss, epoch)
        writer.add_scalar("Reconstruction Loss/train", epoch_recon_loss, epoch)
        writer.add_scalar("PDE Loss/train", epoch_pde_loss, epoch)
        print("Total loss: ", epoch_total_loss, "Reconstruction loss: ", epoch_recon_loss, "PDE loss: ", epoch_pde_loss)

        # save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "linear_model_state_dict": smallLinear.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            }, 
            "trainings/saved_models/" + model_name
        )
        print(f"Saved model, epoch {epoch}.")