"""
This module contains functions to visually inspect synthetic data and evaluate its quality. 
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

def plot_univariate_distributions(obs_df: pd.DataFrame, syn_df: pd.DataFrame, target_folder: str | None) -> None:
    """
    Plot comparisons of the univariate distribution between the observed and synthetic data
    
    :param obs_df: The observed data
    :param syn_df: The synthetic data
    :param target_folder: Folder where images need to be saved

    :return: None
    """
    return None


def plot_spmse(spmse: pd.DataFrame, save_path: str | None, show_plot=True) -> Figure:
    """
    Plot the standardised propensity mean squared error.
    
    :param spmse: The standardised propensity mean squared error values. 
    Should be a 3xN dataframe, where indices 0,1 are the column names and index 2 is the S_pMSE
    :type: Pandas DataFrame
    :param save_path: File name and path to save the image of the plot
    :type: str
    :param show_plot: Boolean index on whether the plot pops up in an interactive window
    :type: Boolean
    :return: None
    """

    if len(spmse.columns) != 3:
        raise ValueError("The dataframe should be of shape 3xN. With index 0 and 1 column names of the data and index 2 the S_pMSE output.") 
    
    bins = [0, 3, 10, 30, 100, np.inf]
    colors = ['rgb(0,0,0)']+ px.colors.sequential.YlOrBr[:5]
    bin_labels = ["MISSING", "(0,3]", "(3,10]", "(10,30]", "(30,100]", '(100,+)']
    #To reviewer, I tried r'$\infty' instead of +, but that broke the code (i.e. no error but showed nothing)

    variables = sorted(list(set(spmse.iloc[:, 0])))
    
    matrix = pd.DataFrame(index=variables[::-1], columns=variables, dtype=float)
    matrix_orig = pd.DataFrame(index=variables[::-1], columns=variables, dtype=float)

    spmse["category"] = np.digitize(spmse.iloc[:,2], bins=bins, right=True)

    for _, row in spmse.iterrows():
        var1, var2, spmse_val, cat = row.iloc[0], row.iloc[1], row.iloc[2], row["category"]
        matrix.loc[var1, var2] = cat
        matrix.loc[var2, var1] = cat
        matrix_orig.loc[var1, var2] = spmse_val
        matrix_orig.loc[var2, var1] = spmse_val

    colorscale = []
    n = len(colors)
    for i, color in enumerate(colors):
        colorscale.append([i/n, color])
        colorscale.append([(i+1)/n, color])

    fig = go.Figure(
        data = go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            text=np.round(matrix_orig.values,4),
            texttemplate="%{text}",
            colorscale=colorscale,
            zmin=0,
            zmax=len(colors),
            colorbar=dict(
                tickmode="array",
                tickvals=np.array(range(len(colors)))+0.5,
                ticktext=bin_labels,
                title="S_pMSE bins"
            )
        ),
        layout=dict(
            title="S_pMSE Heatmap",
            title_x=0.5,
            font=dict(family="Arial", size=15),
            width=900,
            height=845,
            yaxis=dict(scaleanchor="x", scaleratio=1) #Make it a Cube
        )
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)

    if show_plot:
        fig.show()
    
    return fig