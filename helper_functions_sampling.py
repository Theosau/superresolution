import numpy as np
import torch
import torch.nn.functional as F


def get_points(map_one, nbackground=2, nboundary=10, nflow=10):
    
    # separate each points
    background_points = (map_one==0).nonzero()
    boundary_points = (map_one==1).nonzero()
    flow_points = (map_one==2).nonzero()
    
    # get ids
    ids_background = np.random.choice(range(len(background_points)), size=nbackground, replace=False)
    ids_boundary = np.random.choice(range(len(boundary_points)), size=nboundary, replace=False)
    ids_flow = np.random.choice(range(len(flow_points)), size=nflow, replace=False)
    
    # concatenate ids
    pts_one = np.concatenate([
        background_points[ids_background],
        boundary_points[ids_boundary],
        flow_points[ids_flow]
    ],
    axis=0)
    
    return pts_one


def get_all_points(map_one):
    
    # separate each points
    ids_background = (map_one==0).nonzero()
    ids_boundary = (map_one==1).nonzero()
    ids_flow = (map_one==2).nonzero()
    
    # concatenate ids
    pts_all = np.concatenate([
        ids_background,
        ids_boundary,
        ids_flow
    ],
    axis=0)
    
    return pts_all


class PointPooling3D(torch.nn.Module):
    """
    Local pooling operation.
    """

    def __init__(self, interpolation='trilinear'):
        super().__init__()
        self.interp_mode = interpolation

    def forward(self, latent_vectors, map_sample, flow_sample, pts):
        # flip the points because grid sample has different coordinate system
        pts_flipped = pts.flip(dims=(-1,))

        # get points latent vectors: num_images x num_features x num_points
        pts_vectors = F.grid_sample(latent_vectors, pts_flipped, align_corners=True)
        pts_vectors = (pts_vectors.squeeze(-2).squeeze(-2)).permute(0, 2, 1)
        
        # get points segmentation label: num_images x 1 x num_points
        pts_maps = F.grid_sample(map_sample, pts_flipped, align_corners=True)
        pts_maps = (pts_maps.squeeze(2).squeeze(2)).permute(0, 2, 1)
        
        # get points velocities from input data: num_images x 3 x num_points
        pts_flows = F.grid_sample(flow_sample, pts_flipped, align_corners=True)
        pts_flows = (pts_flows.squeeze(-2).squeeze(-2)).permute(0, 2, 1)
        
        return pts_vectors, pts_maps, pts_flows