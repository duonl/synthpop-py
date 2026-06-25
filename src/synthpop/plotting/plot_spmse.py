"""
This module contains a function to visually inspect the quality of the multivariate SpMSE. 
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def _make_matrix(df: pd.DataFrame, value_string="S_pMSE") -> pd.DataFrame:
    """
    Makes a Matrix of the 3xN S_pMSE array.

    :param df: Pandas DataFrame, should be 3xN
    :param value_string: either S_pMSE (matrix for text in figure) 
    or category (matrix for color in figure)

    return: NxN matrix
    """

    matrix = df.pivot(index="column1", columns="column2", values=value_string)
    matrix = matrix.combine_first(matrix.T)

    # invert s.t. the diagonal goes from upper-left to lower-right
    return matrix.iloc[::-1]


def _get_colorscale() -> list:
    """"
    Helper function to obtain the colorscale given the predetermined bins
    """

    colors = ['rgb(255,255,255)'] + px.colors.sequential.YlOrBr[:5]

    n = len(colors)
    colorscale = []

    for i, color in enumerate(colors):

        colorscale.append([i / n, color])
        colorscale.append([(i + 1) / n, color])

    return colorscale


def plot_spmse(spmse: pd.DataFrame, save_path: str | None = None, show_plot: bool = True) -> go.Figure:
    """
    Plot the standardised propensity mean squared error.

    :param spmse: The standardised propensity mean squared error values. 
    Should be a 3xN dataframe, where indices 0,1 are the column names and index 2 is the S_pMSE
    :param save_path: File name and path to save the image of the plot. Does not save if None
    :param show_plot: Boolean on whether the plot pops up in an interactive window

    :return: A Plotly Figure containing a heatmap of pairwise S_pMSE values with
    bin-based colouring and S_PMSE values in the bins.

    Example:
    --------
    >>> import pandas as pd
    >>> from synthpop.plotting.plot import plot_spmse
    >>>
    >>> spmse = pd.DataFrame(
    ...     {
    ...         "column1": ["Age", "Age", "Age", "Income", "Income", "Sex"],
    ...         "column2": ["Age", "Income", "Sex", "Income", "Sex", "Sex"],
    ...         "S_pMSE": [12., 2.5, 15.3, 0.0, 45.7, 0.0],
    ...     }
    ... )
    >>>
    >>> print(spmse)
      column1 column2  S_pMSE
    0     Age     Age     12.
    1     Age  Income     2.5
    2     Age     Sex    15.3
    3  Income  Income     0.0
    4  Income     Sex    45.7
    5     Sex     Sex     0.0
    >>>
    >>> fig = plot_spmse(
    ...     spmse=spmse,
    ...     save_path=None,
    ...     show_plot=True,
    ... )
    """

    if not list(spmse.columns) == ['column1', 'column2', 'S_pMSE']:

        raise ValueError(
            "The dataframe should be of shape 3xN.",
            "With index 0 and 1 named column1 and column2 and index 2 S_pMSE")

    spmse = spmse.copy(deep=False)

    bins = [0, 3, 10, 30, 100, np.inf]
    bin_labels = [
        "MISSING", "(0,3]", "(3,10]",
        "(10,30]", "(30,100]", '(100,+)'
    ]

    spmse["category"] = np.digitize(spmse.iloc[:, 2], bins=bins, right=True)

    matrix = _make_matrix(spmse, "category")
    matrix_orig = _make_matrix(spmse, "S_pMSE")

    # Preprocessing for Plotting
    text_matrix = matrix_orig.round(2).astype(str)
    text_matrix = text_matrix.mask(matrix_orig == 0, "MISSING")

    colorscale = _get_colorscale()

    # plotting
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            text=text_matrix.values,
            texttemplate="<b>%{text}</b>",
            colorscale=colorscale,
            zmin=0,
            zmax=len(bins),
            colorbar=dict(
                tickmode="array",
                tickvals=np.array(range(len(bins)))+0.5,
                ticktext=bin_labels,
                title="S_pMSE bins",
                outlinecolor="black",
                outlinewidth=2,
            )
        ),

        layout=dict(
            title="S_pMSE Heatmap",
            title_x=0.5,
            font=dict(family="Arial", size=15),
            width=900,
            height=845,
            yaxis=dict(scaleanchor="x", scaleratio=1)  # Make it a Cube
        )
    )

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)

    if show_plot:

        fig.show()

    return fig
