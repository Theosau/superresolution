import numpy as np
from tqdm import tqdm

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
    vel = - ((dpl/(2*mu))*((xv_rot**2)/2 - (a+b)*xv_rot + (a*b + (b**2)/2)))
    
    # Set walls
    vel[xv_rot>=(2*a+b)]=0
    vel[xv_rot<=b]=0
    return vel

def generate_dataset(nsamples, xmin=0, xmax=10, ymin=0, ymax=10, nvox=64, verbose=True):
    # flow parameters
#     dpl = np.abs((np.random.rand(nsamples)-0.5)*4) # change this parameter
    dpl = np.random.rand(nsamples)/2 + 1
    mu = 1 # assume mu constant as same fluid
    
    # height parameters
    b = (xmax/2)*np.random.rand(nsamples) # just setting a constraint
    a_max = (xmax - b)/2 #- 2*xmax/nvox # have at least one boundary pixel at the top 
    a = a_max*(np.random.rand(nsamples)/8)*0 + 6*xmax/nvox # add this to prevent a thickness of one pixel
    
    # angle parameter
    angles = (np.random.rand(nsamples)-0.5)*180
    
    # generate samples
    samples = np.zeros((nsamples, 1, nvox, nvox)) # the 1 holds for the number of varaibles (u, v, w, p)?
    for i in tqdm(range(nsamples)):
        samples[i, 0] = generate_example(dpl[i], mu, a[i], b[i], xmin, xmax, ymin, ymax, nvox, angles[i])
    
    velocities = np.zeros((samples.shape[0], 3, samples.shape[2], samples.shape[3]))
    velocities[:, [0]] = np.reshape(np.cos(angles*np.pi/180), (-1, 1, 1, 1))*samples # u
    velocities[:, [1]] = np.reshape(np.sin(angles*np.pi/180), (-1, 1, 1, 1))*samples # v
    # w stays at 0 everywhere
    
    if verbose:
        print("Normalizaing samples.")
    # normalize samples (would have to store those means for evaluation)
    # different normalizations, for each sample or across all dataset
    # are those values normally distributed initially?
    means = velocities.mean(axis=(-1, -2, 0), keepdims=True)
    stds = velocities.std(axis=(-1, -2, 0), keepdims=True)
    stds[0, 2, 0, 0] = 1.0
    norm_velocities = (velocities-means)/stds
    
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
    segmentation_maps[:, 0, :-1, :] += 3*edges_bottom
    segmentation_maps[:, 0, :, :-1] += 3*edges_left
    segmentation_maps[:, 0, :, :-1] += 3*edges_right
    segmentation_maps[segmentation_maps>2]=1

    # simply add edges on the side for now - will be change if we want none 90-degrees rotations
    segmentation_maps[:, :, :, 0][segmentation_maps[:, :, :, 0]==2] = 1
    segmentation_maps[:, :, :, -1][segmentation_maps[:, :, :, -1]==2] = 1
    segmentation_maps[:, :, 0, :][segmentation_maps[:, :, 0, :]==2] = 1
    segmentation_maps[:, :, -1, :][segmentation_maps[:, :, -1, :]==2] = 1
    
    # move segmentations to 3D
    segmentation_maps = np.expand_dims(segmentation_maps, axis=-1)
    segmentation_maps = np.tile(segmentation_maps, (1, 1, 1, 1, nvox))
    segmentation_maps[..., 0][segmentation_maps[..., 0]==2]=1
    segmentation_maps[..., -1][segmentation_maps[..., -1]==2]=1
    
    return norm_velocities, segmentation_maps
