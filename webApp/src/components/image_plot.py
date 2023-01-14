import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from . import ids
import numpy as np
import os

def render(app: Dash) -> html.Div:
    image_data = np.load('src/data/example_image.npy')
    fig1 = px.imshow(image_data, title="Input image")
    fig2 = px.imshow(image_data, title="Model Output")
    return html.Div(children=[
        dcc.Graph(figure=fig1, id=ids.IMAGE_PLOT_1, style={'display': 'inline-block'}),
        dcc.Graph(figure=fig2, id=ids.IMAGE_PLOT_2, style={'display': 'inline-block'})
    ])


# html.Div(dcc.Graph(figure=fig), id=ids.IMAGE_PLOT)