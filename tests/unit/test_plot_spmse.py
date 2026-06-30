import re

import numpy as np
import pandas as pd
import pytest

from synthpop.plotting.plot_spmse import (
    _categorise_spmse,
    _make_matrix,
    _make_text_matrix,
    _get_colour_scale,
    _make_heatmap,
    plot_spmse
)


@pytest.fixture
def spmse_df():
    return pd.DataFrame(
        {
            "column1": ["c1", "c1", "c1", "c2", "c2", "c3"],
            "column2": ["c1", "c2", "c3", "c2", "c3", "c3"],
            "S_pMSE":
            [
                0,
                473842.48534952759345,
                12.4598375983543,
                4.0,
                46.485343962786234,
                0.0001,
            ],
        }
    )


# ----- _categorise_spmse test -----


def test_categorise_spmse_correct_output(spmse_df):
    """
    Test that checks if the S_pMSE is correctly binned
    """
    bins = [0, 3, 10, 30, 100, np.inf]

    spmse_df.loc[1, "S_pMSE"] = 3  # specifically make a boundary condition

    spmse = _categorise_spmse(spmse_df, bins)

    expected = pd.Series([0, 1, 3, 2, 4, 1], name="category")

    pd.testing.assert_series_equal(
        spmse["category"], expected
    )


# ----- _make_matrix test -----


def test_make_matrix_creates_symmetric_matrix():
    """
    Test that checks if the 3xN dataframe correctly makes a matrix
    """
    df = pd.DataFrame(
        {
            "column1": ["A", "A", "A", "B", "B", "C"],
            "column2": ["A", "B", "C", "B", "C", "C"],
            "S_pMSE": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    result = _make_matrix(df)

    expected = pd.DataFrame(
        [[3.0, 5.0, 6.0],
         [2.0, 4.0, 5.0],
         [1.0, 2.0, 3.0]],
        index=["C", "B", "A"],
        columns=["A", "B", "C"],
    )

    pd.testing.assert_frame_equal(result, expected)


# ----- _make_text_matrix tests -----


def test_make_text_matrix_test():
    """
    Test that checks the text matrix
    """
    matrix = pd.DataFrame([
        [12.4598375983543, 46.485343962786234, 0.0001],
        [473842.48534952759345, 4.0, 46.485343962786234],
        [0., 473842.48534952759345,  12.4598375983543]
    ])

    matrix = _make_text_matrix(matrix)

    output = pd.DataFrame(
        [
            ['12.46', '46.49', '0.0'],
            ['473842.49', '4.0', '46.49'],
            ['CONSTANT VARIABLE', '473842.49', '12.46']
        ]
    )

    pd.testing.assert_frame_equal(output, matrix)


# ----- _get_colour_scale tests -----

def test_get_colour_scale_structure():
    """
    Test that checks the colour scale
    """
    colour_scale = _get_colour_scale()

    colours = [
        'rgb(255,255,255)',
        'rgb(255,255,229)',
        'rgb(255,247,188)',
        'rgb(254,227,145)',
        'rgb(254,196,79)',
        'rgb(254,153,41)'
    ]

    assert len(colour_scale) == 2 * len(colours)

    assert colour_scale[0] == [0.0, colours[0]]
    assert colour_scale[-1] == [1.0, colours[-1]]

    for i, colour in enumerate(colours):
        assert colour_scale[2 * i] == [i / len(colours), colour]
        assert colour_scale[2 * i + 1] == [(i + 1) / len(colours), colour]


# ----- _make_heatmap tests -----

@pytest.fixture
def heatmap_inputs():
    matrix = pd.DataFrame(
        [
            [3., 4., 1.],
            [5., 2., 4.],
            [0., 5., 3.]
        ],
        index=["c3", "c2", "c1"],
        columns=["c1", "c2", "c3"],
    )

    text_matrix = pd.DataFrame(
        [
            ["12.46", "46.49", "0.0"],
            ["473842.49", "4.0", "46.49"],
            ["CONSTANT VARIABLE", "473842.49", "12.46"],
        ],
        index=matrix.index,
        columns=matrix.columns,
    )

    bins = [0, 3, 10, 30, 100, np.inf]

    bin_labels = [
        "CONSTANT VARIABLE",
        "(0,3]",
        "(3,10]",
        "(10,30]",
        "(30,100]",
        "(100,+)",
    ]

    colour_scale = [
        [0.0, 'rgb(255,255,255)'],
        [0.16666666666666666, 'rgb(255,255,255)'],
        [0.16666666666666666, 'rgb(255,255,229)'],
        [0.3333333333333333, 'rgb(255,255,229)'],
        [0.3333333333333333, 'rgb(255,247,188)'],
        [0.5, 'rgb(255,247,188)'],
        [0.5, 'rgb(254,227,145)'],
        [0.6666666666666666, 'rgb(254,227,145)'],
        [0.6666666666666666, 'rgb(254,196,79)'],
        [0.8333333333333334, 'rgb(254,196,79)'],
        [0.8333333333333334, 'rgb(254,153,41)'],
        [1.0, 'rgb(254,153,41)']
    ]

    return matrix, text_matrix, colour_scale, bins, bin_labels


def test_make_heatmap_returns_figure(heatmap_inputs):
    """
    Test if a plotly figure is returned
    """
    matrix, text_matrix, colour_scale, bins, bin_labels = heatmap_inputs

    fig = _make_heatmap(
        matrix,
        text_matrix,
        colour_scale,
        bins,
        bin_labels,
    )

    assert fig.data[0].type == "heatmap"


def test_make_heatmap_data(heatmap_inputs):
    """
    Test that the heatmap contains the expected data.
    """

    matrix, text_matrix, colour_scale, bins, bin_labels = heatmap_inputs

    fig = _make_heatmap(
        matrix,
        text_matrix,
        colour_scale,
        bins,
        bin_labels,
    )

    heatmap = fig.data[0]

    assert heatmap.type == "heatmap"
    assert np.shape(heatmap.z) == (3, 3)

    np.testing.assert_array_equal(heatmap.z, matrix.values)
    np.testing.assert_array_equal(heatmap.x, matrix.columns)
    np.testing.assert_array_equal(heatmap.y, matrix.index)
    np.testing.assert_array_equal(heatmap.text, text_matrix.values)


def test_make_heatmap_layout(heatmap_inputs):
    """
    Test that the layout is configured correctly.
    """

    matrix, text_matrix, colour_scale, bins, bin_labels = heatmap_inputs

    fig = _make_heatmap(
        matrix,
        text_matrix,
        colour_scale,
        bins,
        bin_labels,
    )

    assert fig.layout.title.text == "S_pMSE Heatmap"
    assert fig.layout.title.x == 0.5
    assert fig.layout.width == 986
    assert fig.layout.height == 850
    assert fig.layout.xaxis.side == "top"
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_make_heatmap_colourbar(heatmap_inputs):
    """
    Test that the colourbar is configured correctly.
    """

    matrix, text_matrix, colour_scale, bins, bin_labels = heatmap_inputs

    fig = _make_heatmap(
        matrix,
        text_matrix,
        colour_scale,
        bins,
        bin_labels,
    )

    colorbar = fig.data[0].colorbar

    assert list(colorbar.ticktext) == bin_labels

    np.testing.assert_array_equal(
        colorbar.tickvals,
        np.arange(len(bins)) + 0.5,
    )

    assert colorbar.title.text == "S_pMSE bins"


def test_make_heatmap_colour_scale(heatmap_inputs):
    """
    Test that the expected colour scale is used.
    """

    matrix, text_matrix, colour_scale, bins, bin_labels = heatmap_inputs

    fig = _make_heatmap(
        matrix,
        text_matrix,
        colour_scale,
        bins,
        bin_labels,
    )

    assert fig.data[0].colorscale == tuple(map(tuple, colour_scale))


def test_make_heatmap_hovertemplate(heatmap_inputs):
    """
    Test that the expected hover template is used.
    """

    matrix, text_matrix, colour_scale, bins, bin_labels = heatmap_inputs

    fig = _make_heatmap(
        matrix,
        text_matrix,
        colour_scale,
        bins,
        bin_labels,
    )

    assert "%{x}" in fig.data[0].hovertemplate
    assert "%{y}:" in fig.data[0].hovertemplate
    assert "%{text}" in fig.data[0].hovertemplate

# ----- plot_spmse tests -----


@pytest.mark.parametrize(
    "df, match",
    [
        (
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": [2],
                    "column3": ["c3"],
                    "column4": [2]
                }
            ),
            "the columns ['column1', 'column2', 'S_pMSE']"
        ),  # Can only be exactly 3 columns

        (
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "spmse": [1.0]
                }
            ),
            "the columns ['column1', 'column2', 'S_pMSE']"
        ),  # Wrong capitalisation

        (
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": ["c1"],
                    "S_pMSE": [np.nan]
                },
            ),
            "The S_pMSE dataframe must not contain missing value"
        ),  # contains nan

        (
            np.array(['This', 'is', 'not', 'a', 'dataframe'])
        ),
        "The S_pMSE data should be a pandas DataFrame"
        # not a dataframe
    ],
)
def test_input_errors(df, match):
    """
    Test that invalid column names raise a ValueError.
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        plot_spmse(df, None, False)


def test_save_image(monkeypatch, tmp_path, spmse_df):
    """
    Test that write_image is called when save_path is given.
    """

    called = False

    def fake_write_image(*args, **kwargs):
        nonlocal called
        called = True

    from plotly.graph_objects import Figure
    monkeypatch.setattr(Figure, "write_image", fake_write_image)

    plot_spmse(spmse_df, str(tmp_path), False)

    assert called


def test_show_not_called(monkeypatch, spmse_df):
    """
    Test that show() is not called when show_plot=False.
    """

    called = False

    def fake_show(*args, **kwargs):
        nonlocal called
        called = True

    from plotly.graph_objects import Figure
    monkeypatch.setattr(Figure, "show", fake_show)

    plot_spmse(spmse_df, None, False)
    assert not called


def test_show_called(monkeypatch, spmse_df):
    """
    Test that show() is called when show_plot=True.
    """

    called = False

    def fake_show(*args, **kwargs):
        nonlocal called
        called = True

    from plotly.graph_objects import Figure
    monkeypatch.setattr(Figure, "show", fake_show)

    plot_spmse(spmse_df, None, True)
    assert called


def test_no_input_change(spmse_df):
    """
    Test if function does not change the input
    """

    original_df = spmse_df.copy(deep=True)
    plot_spmse(spmse_df, None, False)

    pd.testing.assert_frame_equal(spmse_df, original_df)
