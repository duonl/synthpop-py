"""
This module contains a function to visually inspect the quality of the multivariate SpMSE.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def _categorise_spmse(spmse, bins):
    """
    Categorise S_pMSE values into predefined bins.
    
    This function assumes that the pairwise spmse does not return nan.

    :param spmse: 3xN pandas DataFrame
    :param bins: list of bin edges

    :return: 4xN pandas DataFrame with new categorised column
    
    """
    spmse["category"] = np.digitize(spmse['S_pMSE'], bins=bins, right=True)
    return spmse

def _make_matrix(df: pd.DataFrame, value_string="S_pMSE") -> pd.DataFrame:
    """
    Construct a symmetric matrix of the 3xN S_pMSE array.

    :param df: Pandas DataFrame, should be 3xN
    :param value_string: either S_pMSE (matrix for text in figure)
    or category (matrix for colour in figure)

    return: NxN matrix
    """

    matrix = df.pivot(index="column1", columns="column2", values=value_string)
    matrix = matrix.combine_first(matrix.T)
    matrix = matrix.rename_axis(index=None, columns=None)
    # invert s.t. the diagonal goes from upper-left to lower-right
    return matrix.iloc[::-1]

def _make_text_matrix(matrix):
    """
    Convert a numeric matrix into a string matrix for plotting.

    S_pMSE values equal to zero in the original matrix are replaced with
    the string "CONSTANT VARIABLE", representing cases where the S_pMSE
    is undefined because the variable is constant.

    :param matrix: NxN numpy matrix

    return: NxN matrix
    """
    text_matrix = matrix.round(2).astype(str)
    text_matrix = text_matrix.mask(matrix == 0, "CONSTANT VARIABLE")

    return text_matrix

def _get_colour_scale() -> list:
    """"
    Helper function to obtain the colour scale given the predetermined bins
    """

    colours = ['rgb(255,255,255)'] + px.colors.sequential.YlOrBr[:5]

    n = len(colours)
    colour_scale = []

    for i, colour in enumerate(colours):

        colour_scale.append([i / n, colour])
        colour_scale.append([(i + 1) / n, colour])

    return colour_scale

def _make_heatmap(matrix, text_matrix, colour_scale, bins, bin_labels):
    """
    Generate an interactive Plotly heatmap of the categorised S_pMSE matrix.

    :param matrix: pandas DataFrame of the categorised S_pMSE
    :param text_matrix: pandas DataFrame of the same shape as ``matrix``
        containing the text displayed in each heatmap cell (e.g. rounded
        values or "CONSTANT VARIABLE").
    :param colour_scale: Plotly-compatible colour scale applied to the heatmap.
    :param bins: Sequence of bin edges used to categorise the S_pMSE values.
    :param bin_labels: Labels corresponding to the bins, displayed on the
        colour bar.

    :return: A Plotly Figure containing the S_pMSE heatmap.
    """


    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            text=text_matrix.values,
            texttemplate="<b>%{text}</b>",
            hovertemplate=(
                "x: %{x}<br>"
                "y: %{y}<br>"
                "value: %{text}"
                "<extra></extra>"
            ),
            colorscale=colour_scale,
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
            width=986,
            height=850,
            xaxis=dict(side="top"),
            yaxis=dict(scaleanchor="x", scaleratio=1)  # Make it a Cube
        )
    )

    return fig

def plot_spmse(spmse: pd.DataFrame, save_path: str | None = None, show_plot: bool = True) -> go.Figure:
    """
    Plot the standardised propensity mean squared error.

    :param spmse: DataFrame containing the pairwise standardised propensity mean squared error values. The dataframe must contain exactly the columns `['column1', 'column2', 'S_pMSE']` where `column1` and `column2` identify the variable pair and `S_pMSE` contains the corresponding pairwise S_pMSE value. You can obtain this dataframe by running :func:`~synthpop.utility_metrics.spmse.pairwise_spmse`.
    Should be a 3xN dataframe, where indices 0,1 are the column names and index 2 is the S_pMSE
    :param save_path: File directory to save the image of the plot. Does not save if None
    :param show_plot: Whether to display the heatmap interactively using the active Plotly renderer. Default is `True`. In headless environments this parameter should be set to `False`.

    :return: A Plotly Figure containing a heatmap of pairwise S_pMSE values with
    bin-based colouring and S_PMSE values in the bins.

    Example:
    --------
    >>> import pandas as pd
    >>> from synthpop.plotting import plot_spmse
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
            "The S_pMSE dataframe must contain exactly "
            "the columns ['column1', 'column2', 'S_pMSE']."
        )
    if pd.isna(spmse['S_pMSE']).any():
        raise ValueError(
            "The S_pMSE dataframe must not contain missing values"
        )
    spmse = spmse.copy(deep=False)

    bins = [0, 3, 10, 30, 100, np.inf]
    bin_labels = [
        "CONSTANT VARIABLE", "(0,3]", "(3,10]",
        "(10,30]", "(30,100]", '(100,+)'
    ]

    # pairwise_spmse does not return nan
    spmse = _categorise_spmse(spmse, bins)

    #Make matrix for 2D visualisation
    matrix = _make_matrix(spmse, "category")
    matrix_orig = _make_matrix(spmse, "S_pMSE")

    # Preprocessing for Plotting
    text_matrix = _make_text_matrix(matrix_orig)

    colour_scale = _get_colour_scale()

    # plotting
    fig = _make_heatmap(matrix, text_matrix, colour_scale, bins, bin_labels)

    if save_path:

        output_dir = Path(save_path)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir
            / "spmse.pdf"
        )

        fig.write_image(output_file)

    if show_plot:

        fig.show()

    return fig
