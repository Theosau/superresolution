from dash import Dash, html, dcc
from dash_bootstrap_components.themes import BOOTSTRAP
import plotly.express as px
import pandas as pd
import numpy as np
from src.components.layout import create_layout

def main() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "PC-MRI flow denoising and super-resolution"
    app.layout = create_layout(app)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()