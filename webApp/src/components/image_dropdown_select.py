import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from . import ids
import numpy as np


def render(app: Dash) -> html.Div:
    @app.callback(Output(ids.OUTPUT_FILE_DROPDOWN, 'children'),
                Input(ids.FILE_DROPDOWN, 'value'))
    def update_output(content):
        if content is not None:
            # return f'You selected file: {content}'
            path = 'src/data/' + content
            array = np.load(path, allow_pickle=True)
            fig1 = px.imshow(array, title="Input image")
            fig2 = px.imshow(array, title="Model Output")
            return html.Div(children=[
                    dcc.Graph(figure=fig1, id=ids.IMAGE_PLOT_1, style={'display': 'inline-block'}),
                    dcc.Graph(figure=fig2, id=ids.IMAGE_PLOT_2, style={'display': 'inline-block'})
                ],
                id=ids.IMG_CHART
                )

        else:
            return [html.Li("No files yet!")]
    return html.Div(id=ids.IMG_CHART)