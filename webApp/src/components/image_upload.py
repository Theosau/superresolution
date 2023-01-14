import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from . import ids
import numpy as np
import base64

def render(app: Dash) -> html.Div:
    @app.callback(Output(ids.OUTPUT_IMAGE_UPLOAD, 'children'),
                Input(ids.UPLOAD_IMAGE, 'contents'))
    def update_output(content):
        if content is not None:
            decoded = base64.b64decode(content)
            # with open(decoded, 'rb') as f:
            #     decoded = f.read()
            # print(decoded)
            decoded = 'src/data/example_image.npy'
            array = np.load(decoded, allow_pickle=True)
            fig1 = px.imshow(array, title="Input image")
            fig2 = px.imshow(array, title="Model Output")
            return html.Div(children=[
                dcc.Graph(figure=fig1, id=ids.IMAGE_PLOT_1, style={'display': 'inline-block'}),
                dcc.Graph(figure=fig2, id=ids.IMAGE_PLOT_2, style={'display': 'inline-block'})],
                id=ids.IMG_CHART
                )

        else:
            return [html.Li("No files yet!")]
    return html.Div(id=ids.IMG_CHART)


        # image_data = np.load('src/data/example_image.npy')
        # fig1 = px.imshow(image_data, title="Input image")
        # fig2 = px.imshow(image_data, title="Model Output")
        # return html.Div(children=[
        #     dcc.Graph(figure=fig1, id=ids.IMAGE_PLOT_1, style={'display': 'inline-block'}),
        #     dcc.Graph(figure=fig2, id=ids.IMAGE_PLOT_2, style={'display': 'inline-block'})
        # ])

# def parse_contents(contents, filename, date):
#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),

#         # HTML images accept base64 encoded strings in the same format
#         # that is supplied by the upload
#         html.Img(src=contents),
#         html.Hr(),
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])

# def render(app: Dash) -> html.Div:
#     @app.callback(Output(ids.OUTPUT_IMAGE_UPLOAD, 'children'),
#                 Input(ids.UPLOAD_IMAGE, 'contents'),
#                 State(ids.UPLOAD_IMAGE, 'filename'),
#                 State(ids.UPLOAD_IMAGE, 'last_modified'))
#     def update_output(list_of_contents, list_of_names, list_of_dates):
#         if list_of_contents is not None:
#             children = parse_contents(list_of_contents, list_of_names, list_of_dates)
#             return children
#         else:
#             return [html.Li("No files yet!")]
#     return html.Div(id=ids.IMG_CHART)



# html.Div(dcc.Graph(figure=fig), id=ids.IMAGE_PLOT)