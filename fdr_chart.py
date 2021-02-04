from common import *
from plotly.graph_objects import Figure, Heatmap

FDR_COLORS = ['rgb(55, 85, 35)', 'rgb(1, 252, 122)', 'rgb(231, 231, 231)', 'rgb(255, 23, 81)', 'rgb(128, 7, 45)']

def get_fdr_chart(fdr_by_team_gw: DF, fdr_labels_by_team_gw: DF, title: str, fixed_scale=False) -> Figure:
    scale_min = -1 / 3
    scale_tick = 1 / 3
    if fixed_scale:
        FDR_MIN = 1
        FDR_MAX = 5

        fdr_min = fdr_by_team_gw.min().min()
        fdr_max = fdr_by_team_gw.max().max()

        scale_tick = 1 / (fdr_max - fdr_min)
        scale_min = -(fdr_min - FDR_MIN) * scale_tick
        scale_max = (FDR_MAX - fdr_max) * scale_tick

    fdr_color_scale = []
    scale_val = scale_min
    for color in FDR_COLORS:
        if scale_val >= 0 and scale_val <= 1:
            fdr_color_scale += [[scale_val, color]]
        scale_val += scale_tick

    return Figure(layout=dict(title=title),
                     data=Heatmap(
                         z=fdr_by_team_gw,
                         x=fdr_by_team_gw.columns.values,
                         y=fdr_by_team_gw.index.values,
                         text=fdr_labels_by_team_gw,
                         colorscale=fdr_color_scale,
                         hoverinfo='text',
                         hoverongaps=False))