import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.colors as colors
from .. import const
from .. import loaders
from .. import utils

def comp_pekora_spearmanr(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input_fpath:str,
    points_fpath:str,
    chr2_region:str=None,
):

    count_df = loaders.load_3c_data(
        input_fpath,
        chr1_region,
        resolution,
        balancing=balancing,
        chr2_region=chr2_region,
        ret_df=True
    )

    row_ids = count_df[const.ROW_IDS_COLNAME].to_numpy()
    col_ids = count_df[const.COL_IDS_COLNAME].to_numpy()
    counts = count_df[const.COUNTS_COLNAME].to_numpy()
    
    points = np.load(
        points_fpath
    )
    
    D = utils.edm_numpy.compute_D_from_Ps(
        points,
        row_ids,
        col_ids
    )
    
    res = stats.spearmanr(counts, D)
    corr = -res.statistic
    
    #? data_ratio is one because we do not remove explicitly any region
    #? There is no additional mapping/filtering
    data_ratio = 1.0

    #? data_ratio is one because we do not remove explicitly any region
    print(f"Region (Chromosome):{chr1_region}; Spearman R:{corr:.03f}; Data Ratio:{data_ratio:.03f}")
    
def plot_pekora_coor(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input_fpath:str,
    points_fpath:str,
    chr2_region:str=None,
):
    count_df = loaders.load_3c_data(
        input_fpath,
        chr1_region,
        resolution,
        balancing=balancing,
        chr2_region=chr2_region,
        ret_df=True
    )

    row_ids = count_df[const.ROW_IDS_COLNAME].to_numpy()
    col_ids = count_df[const.COL_IDS_COLNAME].to_numpy()
    # counts = count_df[const.COUNTS_COLNAME].to_numpy()
    
    unique_data_ids = np.unique([row_ids, col_ids])
    
    points = np.load(points_fpath)
    points = points[unique_data_ids, :]
    n_points = len(unique_data_ids)
    
    # colorscale = [
    #     [0, "rgba(68, 119, 170, 1)"],  # blue
    #     [1, "rgba(221, 204, 119, 1)"]  # yellow
    # ]
    # color_norm = np.linspace(0, 1, n_points)
    # colors_plotly = [colors.colorscale_to_str(colorscale, x) for x in color_norm]
    # colors = ["#%02x%02x%02x" % (int(255 * i / n_points), 0, 255 - int(255 * i / n_points)) for i in range(n_points)]
    
    from plotly.express.colors import sample_colorscale
    # colorscale_name = 'viridis'
    colorscale_name = 'bluered'
    c = sample_colorscale(colorscale_name, np.arange(n_points)/n_points)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        # mode='markers+lines', 
        # marker_size=3.0,
        # marker_color=c,
        mode='lines', 
        line_width=4.0,
        line_color=c,
        # marker=dict(size=4, color=[colors[i]], 
        # line=dict(color='black', width=0.5))) for i in range(n_points)
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        # zoom=3.0,
    )
    
    # fig.write_image(points_fpath.replace('.npy', '.png'))
    # fig.write_html("test.html")
    fig.write_html(points_fpath.replace('.npy', '.html'))