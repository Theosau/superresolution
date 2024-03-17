import os, random
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, rotate
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from multiprocessing import current_process
from tqdm.contrib.concurrent import process_map

def generate_example(dpl, mu, a, b, xmin, xmax, ymin, ymax, nvox, angle):
    xaxis = np.linspace(xmin, xmax, nvox)
    yaxis = np.linspace(ymin, ymax, nvox)
    xv, yv = np.meshgrid(xaxis, yaxis, indexing='ij')

    # rotation
    if angle>90 or angle<-90:
        angle = angle%180
    if angle<0:
        b = xmax*np.cos(angle*np.pi/180)/2 - b
    if angle>0:
        b = xmax*np.cos(angle*np.pi/180)/2 + b
    
    xv_rot = xv*np.cos(angle*np.pi/180) + yv*np.sin(angle*np.pi/180)
    yv_rot = - xv*np.sin(angle*np.pi/180) + yv*np.cos(angle*np.pi/180)
    vel = ((dpl/mu)*((xv_rot**2)/2 - (a+b)*xv_rot + (a*b + (b**2)/2)))
    # vel = (((xv_rot-b)**2/mu)*((xv_rot**2)/2 - (a+b)*xv_rot + (a*b + (b**2)/2)))
    # vel = ((yv_rot/mu)*((xv_rot**2)/2 - (a+b)*xv_rot + (a*b + (b**2)/2)))

    # Set walls
    vel[xv_rot>=(2*a+b)]=0
    vel[xv_rot<=b]=0
    return vel


def generate_dataset(
        nsamples, 
        xmin=0, 
        xmax=10,
        ymin=0,
        ymax=10,
        nvox=64,
        sdf=False,
        verbose=True,
        flows_in_z=False,
        multthickness = False,
        multpositions = True,
        multangles = True,
    ):
    
    # flow parameters
    dpl = (np.sign(np.random.rand(nsamples) - 0.5)*((np.random.rand(nsamples)*(9/10) + 0.1))/2)
    mu = 1 # assume mu constant as same fluid

    # translation
    if multpositions:
        b = (xmax/2)*np.random.rand(nsamples) # just setting a constraint
    else:
        b = (xmax/2)*np.ones(nsamples)

    # rotation
    angles = (np.random.rand(nsamples)-0.5)*180
    if not(multangles):
        angles *= 0
        
    # thickness
    if multthickness:
        a = np.random.rand(nsamples) + 1
    else:
        a = np.ones(nsamples)*1.7
    
    # generate samples
    samples = np.zeros((nsamples, 1, nvox, nvox)) # the 1 holds for the number of varaibles (u, v, w, p)?
    for i in tqdm(range(nsamples)):
        samples[i, 0] = generate_example(dpl[i], mu, a[i], b[i], xmin, xmax, ymin, ymax, nvox, angles[i])

    velocities = np.zeros((samples.shape[0], 3, samples.shape[2], samples.shape[3]))
    velocities[:, [0]] = np.reshape(-np.sin(angles*np.pi/180), (-1, 1, 1, 1))*samples # u
    velocities[:, [1]] = np.reshape(np.cos(angles*np.pi/180), (-1, 1, 1, 1))*samples # v
    # w stays at 0 everywhere

    if verbose:
        print("Normalizing samples.")
    # normalize the velocities, each with it's own velocity scale
    velocity_scale = np.max(np.abs(samples), axis=(-1, -2, -3), keepdims=True)
    velocity_scale_for_norm = np.max(np.abs(samples), axis=(-1, -2, -3, -4), keepdims=True) # all with the same scale
    # velocity_scale_for_norm = np.std(samples, axis=(-1, -2, -3, -4), keepdims=True)
    # norm_velocities = velocities/velocity_scale
    norm_velocities = velocities/velocity_scale_for_norm

    # move velocities to 3D
    norm_velocities = np.expand_dims(norm_velocities, axis=-1)
    norm_velocities = np.tile(norm_velocities, (1, 1, 1, 1, nvox))
    
    if verbose:
        print("Creating segmentation maps.")
    segmentation_maps = np.zeros((nsamples, 1, nvox, nvox))
    segmentation_maps[samples!=0] = 2 # points where there is some flow

    # find the edges by computing the gradient
    edges_top = np.minimum(segmentation_maps[:, 0, :-1, :] - segmentation_maps[:, 0, 1:, :], np.zeros_like(segmentation_maps[:, 0, :-1, :]))/(-2)
    edges_bottom = np.minimum(segmentation_maps[:, 0, 1:, :] - segmentation_maps[:, 0, :-1, :], np.zeros_like(segmentation_maps[:, 0, :-1, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_left = np.minimum(segmentation_maps[:, 0, :, :-1] - segmentation_maps[:, 0, :, 1:], np.zeros_like(segmentation_maps[:, 0, :, :-1]))/(-2)
    edges_right = np.minimum(segmentation_maps[:, 0, :, 1:] - segmentation_maps[:, 0, :, :-1], np.zeros_like(segmentation_maps[:, 0, :, :-1]))/(-2) # divide by -2 to get the desired 1 for bounday points

    # add edges to top and bottom
    segmentation_maps[:, 0, :-1, :] += 3*edges_top 
    segmentation_maps[:, 0, 1:, :] += 3*edges_bottom
    segmentation_maps[:, 0, :, :-1] += 3*edges_left
    segmentation_maps[:, 0, :, 1:] += 3*edges_right
    segmentation_maps[segmentation_maps>2]=1

    # simply add edges on the side for now - will be change if we want none 90-degrees rotations
    # comment out to remove the side edges as boundaries
    # because we do not have there exact value and assume that the reconstruction of data
    # points within the flow will be sufficient to recover the information
    # segmentation_maps[:, :, :, 0][segmentation_maps[:, :, :, 0]==2] = 1
    # segmentation_maps[:, :, :, -1][segmentation_maps[:, :, :, -1]==2] = 1
    # segmentation_maps[:, :, 0, :][segmentation_maps[:, :, 0, :]==2] = 1
    # segmentation_maps[:, :, -1, :][segmentation_maps[:, :, -1, :]==2] = 1

    # move segmentations to 3D
    segmentation_maps = np.expand_dims(segmentation_maps, axis=-1)
    segmentation_maps = np.tile(segmentation_maps, (1, 1, 1, 1, nvox))
    # segmentation_maps[..., 0][segmentation_maps[..., 0]==2]=1    # boundary label for flow points in z direction
    # segmentation_maps[..., -1][segmentation_maps[..., -1]==2]=1

    if flows_in_z:
        # making half of them in the z direction
        half = len(segmentation_maps)//2
        segmentation_maps[half:] = np.swapaxes(segmentation_maps[half:], 3, 4)
        norm_velocities[half:] = np.swapaxes(norm_velocities[half:], 3, 4)
        norm_velocities[half:, [1]], norm_velocities[half:, [2]] = norm_velocities[half:, [2]], norm_velocities[half:, [1]]
    
    if verbose:
        print("Filtering out samples with not enough points in boundary and flow")
    # Set the minimum number of ones and twos required
    min_ones = 50
    min_twos = 50

    # Find the indices of ones and twos
    ones_indices = (segmentation_maps == 1).nonzero()
    twos_indices = (segmentation_maps == 2).nonzero()
    
    # Count the number of ones and twos for each row
    ones_counts = np.bincount(ones_indices[0], minlength=segmentation_maps.shape[0])
    twos_counts = np.bincount(twos_indices[0], minlength=segmentation_maps.shape[0])

    # Find the rows that meet the condition
    valid_rows = (ones_counts >= min_ones) & (twos_counts >= min_twos)

    # Create a new array with only the rows that meet the condition
    filtered_maps = segmentation_maps[valid_rows]

    print("Original shape:", segmentation_maps.shape)
    print("Filtered shape:", filtered_maps.shape)

    if sdf:
        sdfs = deepcopy(segmentation_maps)
        sdfs[sdfs==2]=1
        sdfs = signed_distance_field_batch(sdfs.astype(int))
        # sdfs_norm = (sdfs - np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True))/(np.max(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True) - np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True))
        return norm_velocities, segmentation_maps, velocity_scale.squeeze(), sdfs, velocity_scale_for_norm

    return norm_velocities, segmentation_maps, velocity_scale.squeeze(), velocity_scale_for_norm


def generate_pipe_flow_dataset(
        nsamples, 
        xmin=0, 
        xmax=10,
        ymin=0,
        ymax=10,
        nvox=64,
        sdf=False,
        verbose=True,
        add_flows_z=False,
        multthickness = False,
        multpositions = True,
        multangles = False
    ):

    xs = np.linspace(xmin, xmax, nvox)
    ys = np.linspace(xmin, xmax, nvox)
    zs = np.linspace(xmin, xmax, nvox)
    
    xv, _, zv = np.meshgrid(xs, ys, zs, indexing='ij')
    
    # dpl = np.sign(np.random.rand(nsamples) - 0.5)*((np.random.rand(nsamples)*(4/10) + 0.6))
    dpl = np.sign(np.random.rand(nsamples) - 0.5)*((np.random.rand(nsamples)*(9/10) + 0.1))/3 # that's the real new one
    mu  = 1

    if multthickness:
        R = np.random.rand(nsamples)*2 + 2
    else:
        R = np.ones(nsamples)*3

    if multpositions:
        z0 = np.random.rand(nsamples)*4 + 3
    else:
        z0 = np.ones(nsamples)*5

    if multangles:
        angles = (np.random.rand(nsamples).reshape(-1, 1)-0.5)*150
    else:
        angles = np.zeros((nsamples, 1))

    # get centers of rotation
    ys_all = np.repeat(np.expand_dims(ys, axis=0), nsamples, axis=0)
    # center_rot =  (ys_all/1)*np.tan(angles*np.pi/180) + np.random.rand(nsamples).reshape(-1, 1) #+ 3 
    center_rot =  ys_all*np.tan(angles*np.pi/180) + 5*np.random.rand(nsamples).reshape(-1, 1) # that's the real new one
    del ys_all
    
    center_rot = np.expand_dims(center_rot, axis=(1,-1))
    center_rot = np.repeat(np.repeat(center_rot, 64, axis=1), 64, axis=-1)
    
    us = np.zeros((nsamples, nvox, nvox, nvox))
    mask = np.zeros((nsamples, nvox, nvox, nvox), dtype=bool)

    for i in tqdm(range(nsamples)):
        us[i] = dpl[i]/(4*mu) * (((xv-center_rot[i])**2 + (zv-z0[i])**2) - R[i]**2)
        mask[[i]] = ((xv-center_rot[i])**2 + (zv-z0[i])**2) > R[i]**2
    
    us[mask]=0

    # cleanup for memory
    del mask
    del center_rot
    del R
    del z0
    del xv
    del zv

    # get directional velocities from magnitude
    velocities = np.zeros((us.shape[0], 3, us.shape[1], us.shape[2], us.shape[3]))
    velocities[:, 0] = np.reshape(np.sin(angles*np.pi/180), (-1, 1, 1, 1))*us # u
    velocities[:, 1] = np.reshape(np.cos(angles*np.pi/180), (-1, 1, 1, 1))*us # v

    if verbose:
        print("Normalizing samples.")
    # normalize the velocities, each with it's own velocity scale
    velocity_scale = np.max(np.abs(velocities), axis=(-1, -2, -3, -4), keepdims=True)
    velocity_scale_for_norm = np.max(np.abs(velocities), axis=(-1, -2, -3, -4, -5), keepdims=True) # all with the same scale
    norm_velocities = velocities/velocity_scale_for_norm

    if verbose:
        print("Creating segmentation maps.")
    segmentation_maps = np.zeros((nsamples, 1, nvox, nvox, nvox))
    segmentation_maps[np.expand_dims(us, axis=1)!=0] = 2 # points where there is some flow

    # find the edges by computing the gradient
    edges_top = np.minimum(segmentation_maps[:, 0, :-1, :, :] - segmentation_maps[:, 0, 1:, :, :], np.zeros_like(segmentation_maps[:, 0, :-1, :, :]))/(-2)
    edges_bottom = np.minimum(segmentation_maps[:, 0, 1:, :, :] - segmentation_maps[:, 0, :-1, :, :], np.zeros_like(segmentation_maps[:, 0, :-1, :, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_left = np.minimum(segmentation_maps[:, 0, :, :-1, :] - segmentation_maps[:, 0, :, 1:, :], np.zeros_like(segmentation_maps[:, 0, :, :-1, :]))/(-2)
    edges_right = np.minimum(segmentation_maps[:, 0, :, 1:, :] - segmentation_maps[:, 0, :, :-1, :], np.zeros_like(segmentation_maps[:, 0, :, :-1, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_front = np.minimum(segmentation_maps[:, 0, :, :, :-1] - segmentation_maps[:, 0, :, :, 1:], np.zeros_like(segmentation_maps[:, 0, :, :, :-1]))/(-2)
    edges_back = np.minimum(segmentation_maps[:, 0, :, :, 1:] - segmentation_maps[:, 0, :, :, :-1], np.zeros_like(segmentation_maps[:, 0, :, :, :-1]))/(-2) # divide by -2 to get the desired 1 for bounday points

    # add edges to top and bottom
    segmentation_maps[:, 0, :-1, :, :] += 3*edges_top 
    segmentation_maps[:, 0, 1:, :, :] += 3*edges_bottom
    segmentation_maps[:, 0, :, :-1, :] += 3*edges_left
    segmentation_maps[:, 0, :, 1:, :] += 3*edges_right
    segmentation_maps[:, 0, :, :, :-1] += 3*edges_front
    segmentation_maps[:, 0, :, :, 1:] += 3*edges_back
    
    segmentation_maps[segmentation_maps>2]=1

    # simply add edges on the side for now - will be change if we want none 90-degrees rotations
    # comment out to remove the side edges as boundaries
    # because we do not have there exact value and assume that the reconstruction of data
    # points within the flow will be sufficient to recover the information
    segmentation_maps[:, :, :, 0][segmentation_maps[:, :, :, 0, :]==2] = 1
    segmentation_maps[:, :, :, -1][segmentation_maps[:, :, :, -1, :]==2] = 1
    segmentation_maps[:, :, 0, :][segmentation_maps[:, :, 0, :, :]==2] = 1
    segmentation_maps[:, :, -1, :][segmentation_maps[:, :, -1, :, :]==2] = 1
    segmentation_maps[..., 0][segmentation_maps[..., 0]==2]=1    # boundary label for flow points in z direction
    segmentation_maps[..., -1][segmentation_maps[..., -1]==2]=1
    
    if verbose:
        print("Filtering out samples with not enough points in boundary and flow")
    # Set the minimum number of ones and twos required
    min_ones = 50
    min_twos = 50

    # Find the indices of ones and twos
    ones_indices = (segmentation_maps == 1).nonzero()
    twos_indices = (segmentation_maps == 2).nonzero()
    
    # Count the number of ones and twos for each row
    ones_counts = np.bincount(ones_indices[0], minlength=segmentation_maps.shape[0])
    twos_counts = np.bincount(twos_indices[0], minlength=segmentation_maps.shape[0])

    # Find the rows that meet the condition
    valid_rows = (ones_counts >= min_ones) & (twos_counts >= min_twos)

    # Create a new array with only the rows that meet the condition
    filtered_maps = segmentation_maps[valid_rows]

    print("Original shape:", segmentation_maps.shape)
    print("Filtered shape:", filtered_maps.shape)

    if sdf:
        sdfs = deepcopy(segmentation_maps)
        sdfs[sdfs==2]=1
        sdfs = signed_distance_field_batch(sdfs.astype(int))
        sdfs_norm = (sdfs - np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True))/(np.max(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True) - np.min(sdfs, axis=(-1, -2, -3, -4, -5), keepdims=True))
        return norm_velocities, segmentation_maps, velocity_scale.squeeze(), sdfs_norm, velocity_scale_for_norm

    return norm_velocities, segmentation_maps, velocity_scale.squeeze(), velocity_scale_for_norm   


def generate_stenosis(nvox=64, verbose=True, sdf=True, stenosis='all'):
    if stenosis=='all':
        stenosis_samples = np.load('../data/stenosis/stenosis_samples_f0_3_5.npy')
    else:
        stenosis_samples = np.load('../data/stenosis/stenosis_samples.npy')
    samples = np.sqrt(np.sum(stenosis_samples[:, :3, :, :]**2, axis=1, keepdims=True))
    velocities = stenosis_samples[:, :3, :, :, :]
    pressures = stenosis_samples[:, [3], :, :, :]

    if verbose:
        print("Normalizing samples.")
    # normalize the velocities, each with it's own velocity scale
    velocity_scale = np.max(np.abs(samples), axis=(-1, -2, -3), keepdims=True)
    velocity_scale_for_norm = np.max(np.abs(samples), axis=(-1, -2, -3, -4), keepdims=True) # all with the same scale
    norm_velocities = velocities/velocity_scale_for_norm

    # move velocities to 3D
    norm_velocities = np.expand_dims(norm_velocities, axis=-1)
    norm_velocities = np.tile(norm_velocities, (1, 1, 1, 1, nvox))
    
    if verbose:
        print("Creating segmentation maps.")
    segmentation_maps = np.zeros((samples.shape[0], 1, nvox, nvox))
    segmentation_maps[samples!=0] = 2 # points where there is some flow

    # find the edges by computing the gradient
    edges_top = np.minimum(segmentation_maps[:, 0, :-1, :] - segmentation_maps[:, 0, 1:, :], np.zeros_like(segmentation_maps[:, 0, :-1, :]))/(-2)
    edges_bottom = np.minimum(segmentation_maps[:, 0, 1:, :] - segmentation_maps[:, 0, :-1, :], np.zeros_like(segmentation_maps[:, 0, :-1, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_left = np.minimum(segmentation_maps[:, 0, :, :-1] - segmentation_maps[:, 0, :, 1:], np.zeros_like(segmentation_maps[:, 0, :, :-1]))/(-2)
    edges_right = np.minimum(segmentation_maps[:, 0, :, 1:] - segmentation_maps[:, 0, :, :-1], np.zeros_like(segmentation_maps[:, 0, :, :-1]))/(-2) # divide by -2 to get the desired 1 for bounday points

    # add edges to top and bottom
    segmentation_maps[:, 0, :-1, :] += 3*edges_top 
    segmentation_maps[:, 0, 1:, :] += 3*edges_bottom
    segmentation_maps[:, 0, :, :-1] += 3*edges_left
    segmentation_maps[:, 0, :, 1:] += 3*edges_right
    segmentation_maps[segmentation_maps>2]=1

    # move segmentations to 3D
    segmentation_maps = np.expand_dims(segmentation_maps, axis=-1)
    segmentation_maps = np.tile(segmentation_maps, (1, 1, 1, 1, nvox))
    
    if sdf:
        sdfs = deepcopy(segmentation_maps)
        sdfs[sdfs==2]=1
        sdfs[:, :, 0, :, :] = 0 # setting top to 0 to prevent artefact
        sdfs = signed_distance_field_batch(sdfs.astype(int))
        return norm_velocities, segmentation_maps, velocity_scale.squeeze(), sdfs, velocity_scale_for_norm

    return norm_velocities, segmentation_maps, velocity_scale.squeeze(), velocity_scale_for_norm


def generate_3dstenosis(nvox=64, verbose=True, sdf=True, transforms=False, data_aug=2, mode="train"):
    velocities = np.load('../data/3dstenosis/3dstenosis.npy')
    if mode=="eval":
        # Set the parameters
        lower_bound = 0
        upper_bound = len(velocities)-1
        # Generate random integers without replacement
        # selected_integers = np.random.choice(np.arange(lower_bound, upper_bound + 1), 20, replace=False)
        # selected_integers = [307, 191, 199, 165, 208, 88, 296, 212, 44, 138, 101, 195, 204, 2, 216, 113, 254, 16, 116, 315]
        selected_integers = [147, 165, 189, 133, 215, 222, 136,   7,  12,  39, 254, 173,  18,
            29,  19,   3, 157,   9, 132, 146,  36, 199, 111, 240, 130, 243,
            168,  53, 186,  99, 123,  71, 160, 125,  74, 306, 214, 253, 140,
            13, 282, 291, 230,  89,  57,   6, 247,   0,  94, 181, 284,  25,
            268, 114, 226,  90,  60, 172,  79, 235, 113, 278, 180, 112, 143,
            127, 193,  11, 308, 252, 229,  32,  30, 183, 101, 1,  48, 188,
            158, 283, 145, 264, 117, 227, 166,  77, 142, 231,  21,  63, 195,
            305, 106, 297, 296,  52,  98, 275,  97, 274
        ]
        velocities = velocities[selected_integers] # select
        # split
        pressures = velocities[:, [3], :, :, :]
        wsses = np.linalg.norm(velocities[:, 4:, :, :, :], axis=1, keepdims=True)
        velocities = velocities[:, :3, :, :, :]
    else:
        velocities = velocities[:, :3, :, :, :]
    if transforms:
        num_processors = cpu_count()
        if mode=="eval":
            ifpw = [(i, f, p, w) for i, (f, p, w) in enumerate(zip(velocities, pressures, wsses))] * data_aug
            result_list = process_map(process_flow_and_fields, ifpw, max_workers=num_processors)
            del ifpw
            f_list, p_list, w_list = zip(*result_list)
            velocities = np.concatenate(f_list, axis=0) if f_list else None
            del f_list
            pressures = np.concatenate(p_list, axis=0) if p_list else None
            del p_list
            wsses = np.concatenate(w_list, axis=0) if w_list else None
            del w_list

        else:
            # Create a list of tuples containing both indices and the actual flow data
            indices_and_flows = [(i, flow) for i, flow in enumerate(velocities)] * data_aug
            result_list = process_map(process_flow, indices_and_flows, max_workers=num_processors) # , chunksize=10
            # Remove None results if any
            result_list = [result for result in result_list if result is not None]
            velocities = np.concatenate(result_list, axis=0) if result_list else None
            del indices_and_flows
        del result_list

    samples = np.sqrt(np.sum(np.square(velocities[:, :, :, :, :]), axis=1, keepdims=True))

    if verbose:
        print("Normalizing samples.")
    # normalize the velocities, each with it's own velocity scale
    velocity_scale = np.max(np.abs(samples), axis=(-1, -2, -3, -4), keepdims=True)
    velocity_scale_for_norm = np.max(np.abs(samples), axis=(-1, -2, -3, -4, -5), keepdims=True) # all with the same scale
    norm_velocities = velocities/velocity_scale_for_norm
    
    if verbose:
        print("Creating segmentation maps.")
    segmentation_maps = np.zeros((samples.shape[0], 1, nvox, nvox, nvox))
    segmentation_maps[samples!=0] = 2 # points where there is some flow
    del samples

    if verbose:
        print("Adding edges.")
    # find the edges by computing the gradient
    edges_top = np.minimum(segmentation_maps[:, 0, :-1, :, :] - segmentation_maps[:, 0, 1:, :, :], np.zeros_like(segmentation_maps[:, 0, :-1, :, :]))/(-2)
    edges_bottom = np.minimum(segmentation_maps[:, 0, 1:, :, :] - segmentation_maps[:, 0, :-1, :, :], np.zeros_like(segmentation_maps[:, 0, :-1, :, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_left = np.minimum(segmentation_maps[:, 0, :, :-1, :] - segmentation_maps[:, 0, :, 1:, :], np.zeros_like(segmentation_maps[:, 0, :, :-1, :]))/(-2)
    edges_right = np.minimum(segmentation_maps[:, 0, :, 1:, :] - segmentation_maps[:, 0, :, :-1, :], np.zeros_like(segmentation_maps[:, 0, :, :-1, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_front = np.minimum(segmentation_maps[:, 0, :, :, :-1] - segmentation_maps[:, 0, :, :, 1:], np.zeros_like(segmentation_maps[:, 0, :, :, :-1]))/(-2)
    edges_back = np.minimum(segmentation_maps[:, 0, :, :, 1:] - segmentation_maps[:, 0, :, :, :-1], np.zeros_like(segmentation_maps[:, 0, :, :, :-1]))/(-2) # divide by -2 to get the desired 1 for bounday points

    if verbose:
        print("Created edges.")
    
    # add edges to top and bottom
    segmentation_maps[:, 0, :-1, :, :] += 3*edges_top
    del edges_top
    segmentation_maps[:, 0, 1:, :, :] += 3*edges_bottom
    del edges_bottom
    segmentation_maps[:, 0, :, :-1, :] += 3*edges_left
    del edges_left
    segmentation_maps[:, 0, :, 1:, :] += 3*edges_right
    del edges_right
    segmentation_maps[:, 0, :, :, :-1] += 3*edges_front
    del edges_front
    segmentation_maps[:, 0, :, :, 1:] += 3*edges_back
    del edges_back

    segmentation_maps[segmentation_maps>2]=1

    if verbose:
        print("Add edges on the side.")
    # simply add edges on the side for now - will be change if we want none 90-degrees rotations
    # comment out to remove the side edges as boundaries
    # because we do not have there exact value and assume that the reconstruction of data
    # points within the flow will be sufficient to recover the information
    segmentation_maps[:, :, :, 0][segmentation_maps[:, :, :, 0, :]==2] = 1
    segmentation_maps[:, :, :, -1][segmentation_maps[:, :, :, -1, :]==2] = 1
    segmentation_maps[:, :, 0, :][segmentation_maps[:, :, 0, :, :]==2] = 1
    segmentation_maps[:, :, -1, :][segmentation_maps[:, :, -1, :, :]==2] = 1
    segmentation_maps[..., 0][segmentation_maps[..., 0]==2]=1    # boundary label for flow points in z direction
    segmentation_maps[..., -1][segmentation_maps[..., -1]==2]=1

    if sdf:
        if verbose:
            print("Generating sfds.")
        sdfs = deepcopy(segmentation_maps)
        sdfs[sdfs==2]=1
        sdfs = signed_distance_field_batch(sdfs.astype(int))
        if mode=="eval":
            return norm_velocities, pressures, wsses, segmentation_maps, velocity_scale.squeeze(), sdfs, velocity_scale_for_norm
        else:
            return norm_velocities, segmentation_maps, velocity_scale.squeeze(), sdfs, velocity_scale_for_norm

    return norm_velocities, segmentation_maps, velocity_scale.squeeze(), velocity_scale_for_norm


def generate_3daneurysm(nvox=64, verbose=True, sdf=True, transforms=False, data_aug=2, mode="train"):
    velocities = np.load('../data/3daneurysm/3daneurysm.npy')
    if mode=="eval":
        # Set the parameters
        lower_bound = 0
        upper_bound = len(velocities)-1
        # Generate random integers without replacement
        # selected_integers = np.random.choice(np.arange(lower_bound, upper_bound + 1), 20, replace=False)
        # selected_integers = [307, 191, 199, 165, 208, 88, 296, 212, 44, 138, 101, 195, 204, 2, 216, 113, 254, 16, 116, 315]
        selected_integers = [147, 165, 189, 133, 215, 222, 136,   7,  12,  39, 254, 173,  18,
            29,  19,   3, 157,   9, 132, 146,  36, 199, 111, 240, 130, 243,
            168,  53, 186,  99, 123,  71, 160, 125,  74, 306, 214, 253, 140,
            13, 282, 291, 230,  89,  57,   6, 247,   0,  94, 181, 284,  25,
            268, 114, 226,  90,  60, 172,  79, 235, 113, 278, 180, 112, 143,
            127, 193,  11, 308, 252, 229,  32,  30, 183, 101, 1,  48, 188,
            158, 283, 145, 264, 117, 227, 166,  77, 142, 231,  21,  63, 195,
            305, 106, 297, 296,  52,  98, 275,  97, 274
        ]
        velocities = velocities[selected_integers] # select
        # split
        pressures = velocities[:, [3], :, :, :]
        wsses = np.linalg.norm(velocities[:, 4:, :, :, :], axis=1, keepdims=True)
        velocities = velocities[:, :3, :, :, :]
    else:
        velocities = velocities[:, :3, :, :, :]

    if transforms:
        num_processors = cpu_count()
        if mode=="eval":
            ifpw = [(i, f, p, w) for i, (f, p, w) in enumerate(zip(velocities, pressures, wsses))] * data_aug
            result_list = process_map(process_flow_and_fields, ifpw, max_workers=num_processors)
            del ifpw
            f_list, p_list, w_list = zip(*result_list)
            velocities = np.concatenate(f_list, axis=0) if f_list else None
            del f_list
            pressures = np.concatenate(p_list, axis=0) if p_list else None
            del p_list
            wsses = np.concatenate(w_list, axis=0) if w_list else None
            del w_list

        else:
            # Create a list of tuples containing both indices and the actual flow data
            indices_and_flows = [(i, flow) for i, flow in enumerate(velocities)] * data_aug
            result_list = process_map(process_flow, indices_and_flows, max_workers=num_processors) # , chunksize=10
            del indices_and_flows
            # Remove None results if any
            result_list = [result for result in result_list if result is not None]
            velocities = np.concatenate(result_list, axis=0) if result_list else None
        del result_list
        
    samples = np.sqrt(np.sum(np.square(velocities[:, :, :, :, :]), axis=1, keepdims=True))

    if verbose:
        print("Normalizing samples.")
    # normalize the velocities, each with it's own velocity scale
    velocity_scale = np.max(np.abs(samples), axis=(-1, -2, -3, -4), keepdims=True)
    velocity_scale_for_norm = np.max(np.abs(samples), axis=(-1, -2, -3, -4, -5), keepdims=True) # all with the same scale
    norm_velocities = velocities/velocity_scale_for_norm
    
    if verbose:
        print("Creating segmentation maps.")
    segmentation_maps = np.zeros((samples.shape[0], 1, nvox, nvox, nvox))
    segmentation_maps[samples!=0] = 2 # points where there is some flow
    del samples

    if verbose:
        print("Adding edges.")
    # find the edges by computing the gradient
    edges_top = np.minimum(segmentation_maps[:, 0, :-1, :, :] - segmentation_maps[:, 0, 1:, :, :], np.zeros_like(segmentation_maps[:, 0, :-1, :, :]))/(-2)
    edges_bottom = np.minimum(segmentation_maps[:, 0, 1:, :, :] - segmentation_maps[:, 0, :-1, :, :], np.zeros_like(segmentation_maps[:, 0, :-1, :, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_left = np.minimum(segmentation_maps[:, 0, :, :-1, :] - segmentation_maps[:, 0, :, 1:, :], np.zeros_like(segmentation_maps[:, 0, :, :-1, :]))/(-2)
    edges_right = np.minimum(segmentation_maps[:, 0, :, 1:, :] - segmentation_maps[:, 0, :, :-1, :], np.zeros_like(segmentation_maps[:, 0, :, :-1, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    edges_front = np.minimum(segmentation_maps[:, 0, :, :, :-1] - segmentation_maps[:, 0, :, :, 1:], np.zeros_like(segmentation_maps[:, 0, :, :, :-1]))/(-2)
    edges_back = np.minimum(segmentation_maps[:, 0, :, :, 1:] - segmentation_maps[:, 0, :, :, :-1], np.zeros_like(segmentation_maps[:, 0, :, :, :-1]))/(-2) # divide by -2 to get the desired 1 for bounday points

    if verbose:
        print("Created edges.")
    
    # add edges to top and bottom
    segmentation_maps[:, 0, :-1, :, :] += 3*edges_top
    del edges_top
    segmentation_maps[:, 0, 1:, :, :] += 3*edges_bottom
    del edges_bottom
    segmentation_maps[:, 0, :, :-1, :] += 3*edges_left
    del edges_left
    segmentation_maps[:, 0, :, 1:, :] += 3*edges_right
    del edges_right
    segmentation_maps[:, 0, :, :, :-1] += 3*edges_front
    del edges_front
    segmentation_maps[:, 0, :, :, 1:] += 3*edges_back
    del edges_back

    segmentation_maps[segmentation_maps>2]=1

    if verbose:
        print("Add edges on the side.")
    # simply add edges on the side for now - will be change if we want none 90-degrees rotations
    # comment out to remove the side edges as boundaries
    # because we do not have there exact value and assume that the reconstruction of data
    # points within the flow will be sufficient to recover the information
    segmentation_maps[:, :, :, 0][segmentation_maps[:, :, :, 0, :]==2] = 1
    segmentation_maps[:, :, :, -1][segmentation_maps[:, :, :, -1, :]==2] = 1
    segmentation_maps[:, :, 0, :][segmentation_maps[:, :, 0, :, :]==2] = 1
    segmentation_maps[:, :, -1, :][segmentation_maps[:, :, -1, :, :]==2] = 1
    segmentation_maps[..., 0][segmentation_maps[..., 0]==2]=1    # boundary label for flow points in z direction
    segmentation_maps[..., -1][segmentation_maps[..., -1]==2]=1

    if sdf:
        if verbose:
            print("Generating sfds.")
        sdfs = deepcopy(segmentation_maps)
        sdfs[sdfs==2]=1
        sdfs = signed_distance_field_batch(sdfs.astype(int))
        if mode=="eval":
            return norm_velocities, pressures, wsses, segmentation_maps, velocity_scale.squeeze(), sdfs, velocity_scale_for_norm
        else:
            return norm_velocities, segmentation_maps, velocity_scale.squeeze(), sdfs, velocity_scale_for_norm

    return norm_velocities, segmentation_maps, velocity_scale.squeeze(), velocity_scale_for_norm

# to create the signed distance fields
def signed_distance_field(binary_volume):
    # Create an empty distance map
    distance_map = np.empty_like(binary_volume, dtype=np.float64)
    # Use the distance transform to calculate the distance to the nearest non-zero voxel
    distance_transform_edt(binary_volume, distances=distance_map)
    # Invert the distance map and negate the values to get the signed distance field
    sdf = distance_map
    
    # reverse 
    distance_map = np.empty_like(binary_volume, dtype=np.float64)
    binary_volume = (binary_volume-1)*(-1) # invert the volume
    distance_transform_edt(binary_volume, distances=distance_map)
    sdf_inversed = distance_map
    
    return sdf-sdf_inversed

def signed_distance_field_batch(binary_volumes):
    with Pool(cpu_count()) as pool:
        sdfs = list(tqdm(pool.imap(signed_distance_field, binary_volumes), total=binary_volumes.shape[0]))
    return np.array(sdfs)

# transforms
def shift_array(array, shifting, axis=1):
    return np.roll(array, shift=shifting, axis=axis)

def pad_with_last_val(array, pad_width):
    pad_before = pad_width[0]
    pad_after = pad_width[1]

    last_before = array[:, :1, :, :] # Take the first element along axis=1
    last_after = array[:, -1:, :, :] # Take the last element along axis=1

    array = np.concatenate([last_before]*pad_before + [array] + [last_after]*pad_after, axis=1)

    last_before = array[:, :, :1, :] # Take the first element along axis=2
    last_after = array[:, :, -1:, :] # Take the last element along axis=2

    array = np.concatenate([last_before]*pad_before + [array] + [last_after]*pad_after, axis=2)
    
    return array

def rotate_flow(padded_array, angle, axes=(1, 2), padding=15):
    rotated_array = rotate(padded_array, angle, axes=axes, reshape=False)[:, padding:-padding, padding:-padding, :]
    
    # rotate velocity fields
    rotated_vector = np.zeros((rotated_array.shape[0], rotated_array.shape[1], rotated_array.shape[2], rotated_array.shape[3]))
    rotated_vector[[0]] = -np.sin(angle*np.pi/180)*rotated_array[[1]] + np.cos(angle*np.pi/180)*rotated_array[[0]]
    rotated_vector[[1]] = np.cos(angle*np.pi/180)*rotated_array[[1]] + np.sin(angle*np.pi/180)*rotated_array[[0]]
    rotated_vector[[2]] = rotated_array[[2]]
    
    return rotated_vector

def rotate_scalar_field(padded_array, angle, axes=(1, 2), padding=15):
    return rotate(padded_array, angle, axes=axes, reshape=False)[:, padding:-padding, padding:-padding, :]

def invert_flow(flow):
    inverted_array = rotate(flow, angle=180, axes=(1, 2), reshape=False)
    inverted_array = inverted_array*np.reshape(np.array([-1, -1, 1]), (-1, 1, 1, 1))
    return inverted_array

def invert_scalar_field(field):
    return rotate(field, angle=180, axes=(1, 2), reshape=False)

def transform_flow(flow, padding=15):
    segmentation_map = compute_seg_map(flow)
    segmentation_map = add_edges(segmentation_map, ybool=False, borders=False)
    if np.random.rand()>0.5:
        flow = invert_flow(flow)
    # shift
    shifting = int((np.random.rand()-0.5)*30) # int((np.random.rand()-0.5)*40)
    flow = shift_array(flow, shifting)
    segmentation_map = shift_array(segmentation_map, shifting)
    # pad
    flow = pad_with_last_val(flow, [padding, padding])
    segmentation_map = pad_with_last_val(segmentation_map, [padding, padding])
    # rotate
    angle = (np.random.rand()-0.5)*180
    flow = rotate_flow(flow, angle)
    segmentation_map = rotate(segmentation_map, angle, axes=(1, 2), reshape=False, order=1)[:, padding:-padding, padding:-padding, :]
    segmentation_map = np.ceil(segmentation_map)
    # remove background from flow based on segmentation map
    flow[np.repeat(segmentation_map, repeats=3, axis=0)!=2]=0
    return flow

def process_flow(data):
    # Set a different numpy random seed for each worker process
    # seed_number = os.getpid() + current_process()._identity[0]
    seed_number = random.randint(0, 2**32 - 1)
    np.random.seed(seed_number)
    # data contains both the index and the actual flow
    index, flow = data
    try:
        return np.expand_dims(transform_flow(flow), axis=0)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def transform_flow_and_fields(flow, pressure, wss, padding=15):
    segmentation_map = compute_seg_map(flow)
    segmentation_map = add_edges(segmentation_map, ybool=False, borders=False)
    # invert
    if np.random.rand()>0.5:
        flow = invert_flow(flow)
        pressure = invert_scalar_field(pressure)
        wss = invert_scalar_field(wss)
    # shift
    shifting = int((np.random.rand()-0.5)*30) # int((np.random.rand()-0.5)*40)
    flow = shift_array(flow, shifting)
    pressure = shift_array(pressure, shifting)
    wss = shift_array(wss, shifting)
    segmentation_map = shift_array(segmentation_map, shifting)
    # pad
    flow = pad_with_last_val(flow, [padding, padding])
    pressure = pad_with_last_val(pressure, [padding, padding])
    wss = pad_with_last_val(wss, [padding, padding])
    segmentation_map = pad_with_last_val(segmentation_map, [padding, padding])
    # rotate
    angle = (np.random.rand()-0.5)*180
    flow = rotate_flow(flow, angle)
    pressure = rotate(pressure, angle, axes=(1, 2), reshape=False, order=0)[:, padding:-padding, padding:-padding, :]
    wss = rotate(wss, angle, axes=(1, 2), reshape=False)[:, padding:-padding, padding:-padding, :]
    segmentation_map = rotate(segmentation_map, angle, axes=(1, 2), reshape=False, order=1)[:, padding:-padding, padding:-padding, :]
    segmentation_map = np.ceil(segmentation_map)
    # remove background from flow based on segmentatino map
    flow[np.repeat(segmentation_map, repeats=3, axis=0)!=2]=0
    pressure[segmentation_map!=2]=0
    wss[segmentation_map!=2]=0
    return flow, pressure, wss
    
def process_flow_and_fields(data):
    # Set a different numpy random seed for each worker process
    # seed_number = os.getpid() + current_process()._identity[0]
    seed_number = random.randint(0, 2**32 - 1)
    np.random.seed(seed_number)
    index, flow, pressure, wss = data
    try:
        flow, pressure, wss = transform_flow_and_fields(flow, pressure, wss)
        flow = np.expand_dims(flow, axis=0)
        pressure = np.expand_dims(pressure, axis=0)
        wss = np.expand_dims(wss, axis=0)
        return flow, pressure, wss
    except Exception as e:
        print(f"Error occurred: {e}")
        return None 
    
def add_edges(segmentation_map, xbool=True, ybool=True, zbool=True, borders=True):
    # find the edges by computing the gradient
    if xbool:
        edges_top = np.minimum(segmentation_map[0, :-1, :, :] - segmentation_map[0, 1:, :, :], np.zeros_like(segmentation_map[0, :-1, :, :]))/(-2)
        edges_bottom = np.minimum(segmentation_map[0, 1:, :, :] - segmentation_map[0, :-1, :, :], np.zeros_like(segmentation_map[0, :-1, :, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    if ybool:
        edges_left = np.minimum(segmentation_map[0, :, :-1, :] - segmentation_map[0, :, 1:, :], np.zeros_like(segmentation_map[0, :, :-1, :]))/(-2)
        edges_right = np.minimum(segmentation_map[0, :, 1:, :] - segmentation_map[0, :, :-1, :], np.zeros_like(segmentation_map[0, :, :-1, :]))/(-2) # divide by -2 to get the desired 1 for bounday points
    if zbool:
        edges_front = np.minimum(segmentation_map[0, :, :, :-1] - segmentation_map[0, :, :, 1:], np.zeros_like(segmentation_map[0, :, :, :-1]))/(-2)
        edges_back = np.minimum(segmentation_map[0, :, :, 1:] - segmentation_map[0, :, :, :-1], np.zeros_like(segmentation_map[0, :, :, :-1]))/(-2) # divide by -2 to get the desired 1 for bounday points
    
    # add edges to top and bottom
    if xbool:
        segmentation_map[0, :-1, :, :] += 3*edges_top
        segmentation_map[0, 1:, :, :] += 3*edges_bottom
    if ybool:
        segmentation_map[0, :, :-1, :] += 3*edges_left
        segmentation_map[0, :, 1:, :] += 3*edges_right
    if zbool:
        segmentation_map[0, :, :, :-1] += 3*edges_front
        segmentation_map[0, :, :, 1:] += 3*edges_back
    
    segmentation_map[segmentation_map>2]=1
    
    if borders:
        segmentation_map[:, :, 0, :][segmentation_map[:, :, 0, :]==2] = 1
        segmentation_map[:, :, -1, :][segmentation_map[:, :, -1, :]==2] = 1
        segmentation_map[:, 0, :, :][segmentation_map[:, 0, :, :]==2] = 1
        segmentation_map[:, -1, :, :][segmentation_map[:, -1, :, :]==2] = 1
        segmentation_map[:, :, :, 0][segmentation_map[:, :, :, 0]==2]=1    # boundary label for flow points in z direction
        segmentation_map[:, :, :, -1][segmentation_map[:, :, :, -1]==2]=1
    
    return segmentation_map

def compute_seg_map(flow): # nvox=64
    sample = np.sqrt(np.sum(np.square(flow), axis=0, keepdims=True))
    segmentation_map = np.zeros((1, flow.shape[-3], flow.shape[-2], flow.shape[-1])) # nvox
    segmentation_map[sample!=0] = 2 # points where there is some flow
    return segmentation_map