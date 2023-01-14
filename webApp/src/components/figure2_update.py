import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from . import ids
import numpy as np


def render(app: Dash) -> html.Div:
    @app.callback(Output(ids.IMAGE_PLOT_2, 'figure'),
                  Input(ids.UPDATE_FIGURE_2, 'n_clicks')
    )
    def update_output():
        array = np.zeros(shape=(64, 64))
        fig2 = px.imshow(array, title="Model Output")
        return html.Div(children=[
                dcc.Graph(figure=fig2, id=ids.IMAGE_PLOT_2, style={'display': 'inline-block'})
            ],
            id=ids.IMG_CHART
            )
    return html.Div(id=ids.IMG_CHART)