# Theophile Sautory - webApp for 4D blodd flow super-resolution

##########################################################################
# Model loading - which will be passed on to another server 
########################################################################## 

import os, yaml
initial_dir = os.getcwd()
os.chdir("/Users/theophilesautory/Documents/BerkeleyPhD/Research/superresolution/")

import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append("/Users/theophilesautory/Documents/BerkeleyPhD/Research/superresolution/")
from data.data_generation import generate_dataset
from helper_functions_sampling import get_all_points, PointPooling3D

from train_script import PoseuilleFlowAnalytic
from networks_models import ConvUNetBis, SmallLinear
from torch.utils.data import DataLoader
import torch
torch.set_num_threads(1)
import numpy as np
from tqdm import tqdm
from loss_functions import PDELoss, ReconLoss

def ravel_multi_index(coords, shape):
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.
    This is a `torch` implementation of `numpy.ravel_multi_index`.
    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.
    Returns:
        The raveled indices, (*,).
    """
    shape = torch.tensor(shape + (1,), dtype=coords.dtype)
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    sol = (coords * coefs).sum(dim=-1)
    return sol

model_name = "current_model"

# loading the model config
with open(f"trainings/config_models/{model_name}.yml") as file:
    config = yaml.safe_load(file)

# setting device and data types
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# nvox = config["nvox"]
# nsamples = 100 #config["nsamples"]
# samples, segmentation_maps = generate_dataset(nsamples=nsamples, nvox=nvox)
# np.save('poiseuille_volume.npy', samples)
# np.save('poiseuille_volume_segmentation.npy', segmentation_maps)
# val_dataset = PoseuilleFlowAnalytic(samples, segmentation_maps)
# batch_size = 1 #config["batch_size"]
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# setup models
model = ConvUNetBis(
    input_size=config["nvox"], 
    channels_in=3, #samples.shape[1], 
    channels_init=config["channels_init"],
    channels_out=config["channels_out"],
)
model = model.to(device=device)
model.eval()
print("There are ", sum(p.numel() for p in model.parameters()), " parameters to train.")

smallLinear = SmallLinear(num_features=model.channels_out+4, num_outputs=4) # + x, y, segmentation label
smallLinear.eval()
print("There are ", sum(p.numel() for p in smallLinear.parameters()), " parameters to train.")

checkpoint = torch.load(f"trainings/saved_models/{model_name}", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
smallLinear.load_state_dict(checkpoint["linear_model_state_dict"])

# point sampler
pp3d = PointPooling3D(interpolation="trilinear")

# epoch_total_loss = 0
# epoch_recon_loss = 0
# epoch_pde_loss = 0

# predictions = torch.from_numpy(np.zeros((len(samples), 4, samples.shape[2], samples.shape[3], samples.shape[4])))
# predictions = predictions.float()

# iterate through the dataset
# for i, (flow_sample, map_sample) in enumerate(val_dataloader):
#     flow_sample = flow_sample.to(device=device, dtype=dtype) # move to device, e.g. GPU
#     map_sample = map_sample.to(device=device, dtype=dtype)

#     # =====================forward======================
#     # compute latent vectors
#     latent_vectors = model(flow_sample)
    
#     # select same number of points per image to sample, unsqueeze at dim 1 to get the shape
#     # batch_size x 1 x num_points x coordinates_size
    
#     pts_ints = torch.cat(
#         [get_all_points(map_one.squeeze()).unsqueeze(0) for map_one in map_sample], 
#         dim=0
#     ).unsqueeze(1).unsqueeze(1)
    
# #     pts_ints = torch.Tensor(np.array([get_all_points(map_one.squeeze()) for map_one in map_sample])).unsqueeze(1).unsqueeze(1)
#     # normalize the points to be in -1, 1 (required by grid_sample)
#     # to have it zoomed, I divide these points here by whatever number
#     pts = (2*(pts_ints/(nvox-1)) - 1)/1
    
#     # features, segmentation maps, flow values
#     pts_vectors, pts_maps, pts_flows = pp3d(latent_vectors, map_sample, flow_sample, pts)
#     pts_locations = pts.squeeze()
#     pts_locations.requires_grad = True # needed for the PDE loss
    
#     # create feature vectors for each sampled point
#     feature_vector = torch.cat([pts_locations, pts_maps, pts_vectors], dim=-1) 
#     feature_vector = feature_vector.reshape((-1, model.channels_out + 4)) # x, y, z, seg inter, features
    
#     # split input features to allow taking separate derivatives
#     inputs = [feature_vector[..., i:i+1] for i in range(feature_vector.shape[-1])]
#     x_ = torch.cat(inputs, axis=-1)
    
#     # forward through linear model
#     outputs_linear = smallLinear(x_)
    
#     # points as integers for indexing
#     pts_ints_l = pts_ints.squeeze().long()

#     for k in tqdm(range(len(pts_ints_l))):
#         pts_one_long = pts_ints_l[k]
#         outputs_linear_one = outputs_linear.reshape(batch_size, 64**3, 4)[k]

#         flat_one_points = ravel_multi_index(pts_one_long, (64, 64, 64))
#         a = torch.zeros((4, 64*64*64), dtype=outputs_linear_one.dtype, requires_grad=False)
#         a[:, flat_one_points] = outputs_linear_one.T
#         predictions[i*batch_size + k] = a.reshape((4, 64, 64, 64))
        
#     # =====================losses======================
#     # get the losses weights for each point
#     num_loss_terms = 2
#     seg_interpolations_rand = pts_maps.reshape((-1, 1))
#     weights = torch.ones((len(seg_interpolations_rand), num_loss_terms), dtype=torch.float)
#     # boundary weights
#     weights[(seg_interpolations_rand>=0.75).squeeze(), :] = torch.Tensor([boundary_pde, boundary_recon])
#     # flow weights
#     weights[(seg_interpolations_rand>=1.25).squeeze(), :] = torch.Tensor([flow_pde, flow_recon])
#     # background weights
#     weights[(seg_interpolations_rand<0.5).squeeze(), :] = torch.Tensor([background_pde, background_recon])
    
#     # compute the loss
#     pde_loss = weights[:, 0]*pde_loss_function.compute_loss(inputs, outputs_linear)
#     recon_loss = weights[:, 1]*recon_loss_function.compute_loss(pts_flows, outputs_linear)
#     loss = torch.mean(pde_loss + recon_loss)
    
#     # update write iteration loss
#     epoch_total_loss += loss.item()
#     epoch_recon_loss += recon_loss.mean().item()
#     epoch_pde_loss += pde_loss.mean().item()

# print("Total loss: ", epoch_total_loss, "Reconstruction loss: ", epoch_recon_loss, "PDE loss: ", epoch_pde_loss)
# # np.save("predictions.npy", predictions.detach().cpu().numpy())

# predictions_np = predictions.detach().cpu().numpy()

os.chdir(initial_dir)
##########################################################################
# Web App
########################################################################## 

from dash import Dash, html, dcc, callback_context
import plotly.express as px
import numpy as np
from src.components import ids
from dash.dependencies import Input, Output, State
# import os

app = Dash(__name__, external_stylesheets=['https://fonts.googleapis.com/css?family=Lato'])

app.layout = html.Div(children=[
    html.H1(children='PC-MRI flow denoising and super-resolution', style={'textAlign': 'center', 'font-family': 'Lato'}),
    dcc.Dropdown(
        id=ids.FILE_DROPDOWN,
        options=[
            {'label': f, 'value': f} for f in os.listdir('src/data/')
        ],
        style={'font-family': 'Lato'},
        placeholder='Select a file',
        value=None
    ),
    dcc.Store(id=ids.FILE_DROPDOWN_STORE, data=None),
    dcc.Graph(id=ids.IMAGE_PLOT_1, style={'display': 'inline-block'}),
    dcc.Graph(id=ids.IMAGE_PLOT_2, style={'display': 'inline-block'}),
    html.Button(
        children=['Denoise'], 
        id=ids.UPDATE_FIGURE_2, 
        n_clicks=0, 
        style={
            'font-family': 'Lato',
            # 'background-color': 'blue',
            # 'color': 'white',
            # 'font-size': '18px',
            # 'padding': '10px',
            # 'border-radius': '5px',
        }),
    html.Div(),
    dcc.Textarea(id=ids.COORDINATES, value="Coordinates: x=[0; 64], [y=0; 64]", style={'width': '50%', 'font-family': 'Lato'}),
    dcc.Store(id=ids.RELAYOUT_DATA_STATE, data=None),
    html.Div(),
    html.Button(children=['Super-resolve'], id=ids.SUPERRESOLVE_FIGURE_1, n_clicks=0, style={'font-family': 'Lato'}),
])

@app.callback(Output(ids.IMAGE_PLOT_1, 'figure'), Input(ids.FILE_DROPDOWN, 'value'))
def update_output(content):
    if content is not None:
        # return f'You selected file: {content}'
        path = 'src/data/' + content
        array = np.load(path, allow_pickle=True)
        # print(array.argmax(axis=2))
        sliced = array[0, 0, ..., 15]
        fig1 = px.imshow(sliced, title="Input image")
        return fig1
    else:
        array = np.zeros(shape=(2,2))
        fig1 = px.imshow(array, title="Input image")
        return fig1

@app.callback(
    Output(ids.IMAGE_PLOT_2, 'figure'), 
    [
        Input(ids.UPDATE_FIGURE_2, 'n_clicks'),
        Input(ids.SUPERRESOLVE_FIGURE_1, 'n_clicks'),
    ],
    [
        State(ids.FILE_DROPDOWN, 'value'),
        State(ids.FILE_DROPDOWN_STORE, 'data'),
        State(ids.IMAGE_PLOT_1, 'relayoutData')
    ]
)
def denoise_input(n_clicks_denoise, n_clicks_superresolve, value, data, relayout_data):
    ctx = callback_context
    print(ctx.triggered[0]['prop_id'])
    data = value # store in browser
    content = value 
    print(content)
    if ctx.triggered[0]['prop_id'] == '.':
        array = np.zeros(shape=(2,2))
        fig1 = px.imshow(array, title="Model Output")

    elif ctx.triggered[0]['prop_id'] == f'{ids.UPDATE_FIGURE_2}.n_clicks':
        path = 'src/data/' + content
        array = torch.from_numpy(np.load(path, allow_pickle=True))[:1, ...]
        path_seg = path.split('.')[0] + '_segmentation.npy'
        segmentation = torch.from_numpy(np.load(path_seg, allow_pickle=True))[:1, ...]

        predictions = torch.from_numpy(np.zeros((len(array), 4, array.shape[2], array.shape[3], array.shape[4])))
        predictions = predictions.float()

        flow_sample = array.to(device=device, dtype=dtype) # move to device, e.g. GPU
        map_sample = segmentation.to(device=device, dtype=dtype)

        # =====================forward======================
        # compute latent vectors
        latent_vectors = model(flow_sample)
        
        # select same number of points per image to sample, unsqueeze at dim 1 to get the shape
        # batch_size x 1 x num_points x coordinates_size
        pts_ints = torch.cat(
            [get_all_points(map_one.squeeze()).unsqueeze(0) for map_one in map_sample], 
            dim=0
        ).unsqueeze(1).unsqueeze(1)
        
        # pts_ints = torch.Tensor(np.array([get_all_points(map_one.squeeze()) for map_one in map_sample])).unsqueeze(1).unsqueeze(1)
        # normalize the points to be in -1, 1 (required by grid_sample)
        # to have it zoomed, I divide these points here by whatever number
        pts = (2*(pts_ints/(array.shape[3]-1)) - 1)/1
        
        # features, segmentation maps, flow values
        pts_vectors, pts_maps, pts_flows = pp3d(latent_vectors, map_sample, flow_sample, pts)
        pts_locations = pts.squeeze()
        pts_locations = pts_locations.unsqueeze(0)
        pts_locations.requires_grad = True # needed for the PDE loss
        
        # create feature vectors for each sampled point
        feature_vector = torch.cat([pts_locations, pts_maps, pts_vectors], dim=-1) 
        feature_vector = feature_vector.reshape((-1, model.channels_out + 4)) # x, y, z, seg inter, features
        
        # split input features to allow taking separate derivatives
        inputs = [feature_vector[..., i:i+1] for i in range(feature_vector.shape[-1])]
        x_ = torch.cat(inputs, axis=-1)
        
        # forward through linear model
        outputs_linear = smallLinear(x_)
        
        # points as integers for indexing
        print(pts_ints.shape)
        pts_ints_l = pts_ints.squeeze().long()
        pts_ints_l = pts_ints_l.unsqueeze(0)
        print(pts_ints_l.shape)

        for k in tqdm(range(len(pts_ints_l))):
            pts_one_long = pts_ints_l[k]
            outputs_linear_one = outputs_linear.reshape(array.shape[0], 64**3, 4)[k]

            flat_one_points = ravel_multi_index(pts_one_long, (64, 64, 64))
            a = torch.zeros((4, 64*64*64), dtype=outputs_linear_one.dtype, requires_grad=False)
            a[:, flat_one_points] = outputs_linear_one.T
            predictions[k] = a.reshape((4, 64, 64, 64))
        

        model_output = predictions.detach().cpu().numpy()[0, 0, ..., 15]
        fig1 = px.imshow(model_output, title="Model Output")

    elif ctx.triggered[0]['prop_id'] == f'{ids.SUPERRESOLVE_FIGURE_1}.n_clicks':
        # array = np.zeros(shape=(2,2))
        # fig1 = px.imshow(array, title="Model Output")

        x_coord_l = relayout_data['xaxis.range[0]']
        x_coord_r = relayout_data['xaxis.range[1]']
        # y_coord_l = relayout_data['yaxis.range[0]']
        # y_coord_r = relayout_data['yaxis.range[1]']
        y_coord_b = relayout_data['yaxis.range[0]']
        y_coord_t = relayout_data['yaxis.range[1]']

        path = 'src/data/' + content
        array = torch.from_numpy(np.load(path, allow_pickle=True))[:1, ...]
        path_seg = path.split('.')[0] + '_segmentation.npy'
        segmentation = torch.from_numpy(np.load(path_seg, allow_pickle=True))[:1, ...]

        predictions = torch.from_numpy(np.zeros((len(array), 4, array.shape[2], array.shape[3], array.shape[4])))
        predictions = predictions.float()

        flow_sample = array.to(device=device, dtype=dtype) # move to device, e.g. GPU
        map_sample = segmentation.to(device=device, dtype=dtype)

        # =====================forward======================
        # compute latent vectors
        latent_vectors = model(flow_sample)
        
        # select same number of points per image to sample, unsqueeze at dim 1 to get the shape
        # batch_size x 1 x num_points x coordinates_size
        pts_ints = torch.cat(
            [get_all_points(map_one.squeeze()).unsqueeze(0) for map_one in map_sample], 
            dim=0
        ).unsqueeze(1).unsqueeze(1)
        
        # pts_ints = torch.Tensor(np.array([get_all_points(map_one.squeeze()) for map_one in map_sample])).unsqueeze(1).unsqueeze(1)
        # normalize the points to be in -1, 1 (required by grid_sample)
        # to have it zoomed, I divide these points here by whatever number
        # pts = (2*(pts_ints/(array.shape[3]-1)) - 1)/1
        # ws_small = torch.tensor([x_coord_l, y_coord_r, 0])
        # ws_large = torch.tensor([x_coord_r, y_coord_l, 63])
        ws_small = torch.tensor([y_coord_t, x_coord_l, 0])
        ws_large = torch.tensor([y_coord_b, x_coord_r, 63])
        ws_small.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ws_large.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pts = ( (ws_large - ws_small) * (pts_ints/(array.shape[3]-1)) + ws_small)
        pts = (2*(pts/(array.shape[3]-1)) - 1)
        
        # features, segmentation maps, flow values
        pts_vectors, pts_maps, pts_flows = pp3d(latent_vectors, map_sample, flow_sample, pts)
        pts_locations = pts.squeeze()
        pts_locations = pts_locations.unsqueeze(0)
        pts_locations.requires_grad = True # needed for the PDE loss
        
        # create feature vectors for each sampled point
        feature_vector = torch.cat([pts_locations, pts_maps, pts_vectors], dim=-1) 
        feature_vector = feature_vector.reshape((-1, model.channels_out + 4)) # x, y, z, seg inter, features
        
        # split input features to allow taking separate derivatives
        inputs = [feature_vector[..., i:i+1] for i in range(feature_vector.shape[-1])]
        x_ = torch.cat(inputs, axis=-1)
        
        # forward through linear model
        outputs_linear = smallLinear(x_)
        
        # points as integers for indexing
        print(pts_ints.shape)
        pts_ints_l = pts_ints.squeeze().long()
        pts_ints_l = pts_ints_l.unsqueeze(0)
        print(pts_ints_l.shape)

        for k in tqdm(range(len(pts_ints_l))):
            pts_one_long = pts_ints_l[k]
            outputs_linear_one = outputs_linear.reshape(array.shape[0], 64**3, 4)[k]

            flat_one_points = ravel_multi_index(pts_one_long, (64, 64, 64))
            a = torch.zeros((4, 64*64*64), dtype=outputs_linear_one.dtype, requires_grad=False)
            a[:, flat_one_points] = outputs_linear_one.T
            predictions[k] = a.reshape((4, 64, 64, 64))
        

        model_output = predictions.detach().cpu().numpy()[0, 0, ..., 15]
        fig1 = px.imshow(model_output, title="Model Output")

    return fig1




@app.callback(Output(ids.COORDINATES, 'value'),
              [
                Input(ids.IMAGE_PLOT_1, 'relayoutData'),
                State(ids.RELAYOUT_DATA_STATE,'data')
              ])
def store_coordinates(relayout_data, relayout_data_state):
    if relayout_data == relayout_data_state or relayout_data == {'autosize': True} or relayout_data == {'xaxis.autorange': True, 'yaxis.autorange': True}:
        return "Coordinates: Vertical=[0; 64], Horizontal=[0; 64]"
    else:
        print(relayout_data)
        x_coord_l = relayout_data['xaxis.range[0]']
        x_coord_r = relayout_data['xaxis.range[1]']
        y_coord_b = relayout_data['yaxis.range[0]']
        y_coord_t = relayout_data['yaxis.range[1]']
        return f"Coordinates: Vetical=[{y_coord_t:.2f}; {y_coord_b:.2f}] Horizontal=[{x_coord_l:.2f}; {x_coord_r:.2f}]"


# @app.callback(
#     Output(ids.IMAGE_PLOT_2, 'figure'), 
#     [
#         Input(ids.SUPERRESOLVE_FIGURE_1, 'n_clicks'),
#         Input(ids.COORDINATES, 'value')
#     ]
# )
# def denoise_input(n_clicks, content):
#     if n_clicks:
#         array = np.zeros(shape=(2,2))
#         fig1 = px.imshow(array, title="Model Output")
#         return fig1

if __name__ == '__main__':
    app.run_server(debug=True)

