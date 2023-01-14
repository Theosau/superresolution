from dash import Dash, html
from . import image_plot, image_upload, image_dropdown_select, figure2_update
import dash_core_components as dcc
from . import ids
import os



def create_layout(app: Dash) -> html.Div:
    return html.Div(
        className="app-div",
        children=[
            html.H1(app.title),
            html.Hr(),
            dcc.Dropdown(
                id=ids.FILE_DROPDOWN,
                options=[
                    {'label': f, 'value': f} for f in os.listdir('src/data/')
                ],
                placeholder='Select a file',
                value=None
            ),
            html.Div(id=ids.OUTPUT_FILE_DROPDOWN),
            html.Button('Update figure 2', id=ids.UPDATE_FIGURE_2, n_clicks=0),
            image_dropdown_select.render(app),
            # figure2_update.render(app),
        ]
    )



# def create_layout(app: Dash) -> html.Div:
#     return html.Div(
#         className="app-div",
#         children=[
#             html.H1(app.title),
#             html.Hr(),
#             dcc.Upload(
#                 id=ids.UPLOAD_IMAGE,
#                 children=html.Div(['''
#                 Upload or drag and image to super-resolve.
#                 '''
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '60px',
#                     'lineHeight': '60px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '10px'
#                 },
#                 # Allow multiple files to be uploaded
#                 multiple=False
#             ),
#             html.Div(id=ids.OUTPUT_IMAGE_UPLOAD),
#             image_dropdown_select.render(app),
#             # image_upload.render(app),
#             # image_plot.render(app),
#         ]
#     )