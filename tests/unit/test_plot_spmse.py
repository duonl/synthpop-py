import numpy as np
import pytest
import pandas as pd
import re

from synthpop.plotting.plot_spmse import (
    _categorise_spmse,
    _make_matrix,
    _get_colourscale,
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
    bins = [0, 3, 10, 30, 100, np.inf]

    spmse_df.loc[1, "S_pMSE"] = 3 # specifically make a boundary condition

    spmse = _categorise_spmse(spmse_df, bins)

    expected = pd.Series([0, 1, 3, 2, 4, 1], name="category")
        
    pd.testing.assert_series_equal(
        spmse["category"], expected
    )
# ----- _make_matrix test -----


def test_make_matrix_creates_symmetric_matrix():
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


# ----- _get_colourscale tests -----

def test_get_colourscale_structure():
    colourscale = _get_colourscale()

    colours = [
        'rgb(255,255,255)',
        'rgb(255,255,229)',
        'rgb(255,247,188)',
        'rgb(254,227,145)',
        'rgb(254,196,79)',
        'rgb(254,153,41)'
    ]

    assert len(colourscale) == 2 * len(colours)

    assert colourscale[0] == [0.0, colours[0]]
    assert colourscale[-1] == [1.0, colours[-1]]

    for i, colour in enumerate(colours):
        assert colourscale[2 * i] == [i / len(colours), colour]
        assert colourscale[2 * i + 1] == [(i + 1) / len(colours), colour]

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
    ],
)
def test_input_errors(df, match):
    """
    Test that invalid column names raise a ValueError.
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        plot_spmse(df, None, False)


def test_returns_figure(spmse_df):
    """
    Test that function returns a Figure.
    """

    fig = plot_spmse(spmse_df, None, False)

    assert fig is not None
    assert len(fig.data) == 1


def test_heatmap_trace(spmse_df):
    """
    Test that the trace is a heatmap.
    """

    fig = plot_spmse(spmse_df, None, False)
    heatmap = fig.data[0]

    assert heatmap.type == "heatmap"


def test_heatmap_shape(spmse_df):
    """
    Test that the heatmap z matrix has correct shape.
    """

    fig = plot_spmse(spmse_df, None, False)
    z = fig.data[0].z

    assert np.shape(z) == (3, 3)


def test_binning(spmse_df):
    """
    Test that binned categories are correct (0-5).
    """

    fig = plot_spmse(spmse_df, None, False)
    z = np.array(fig.data[0].z)

    output = np.array(
        [
            [3., 4., 1.],
            [5., 2., 4.],
            [0., 5., 3.]
        ]
    )

    assert (output == z).all()


def test_axis(spmse_df):
    """
    Test the x and y axis of the data.
    """

    fig = plot_spmse(spmse_df, None, False)
    x = np.array(fig.data[0].x)
    y = np.array(fig.data[0].y)

    assert (x == np.array(['c1', 'c2', 'c3'])).all()
    assert (y == np.array(['c3', 'c2', 'c1'])).all()


def test_text(spmse_df):
    """
    Test that cells with 0 S_pMSE are shown as 'CONSTANT VARIABLE'.
    """

    fig = plot_spmse(spmse_df, None, False)
    text = np.array(fig.data[0].text)

    output = np.array(
        [
            ['12.46', '46.49', '0.0'],
            ['473842.49', '4.0', '46.49'],
            ['CONSTANT VARIABLE', '473842.49', '12.46']
        ]
    )

    assert (output == text).all()


def test_colourbar_labels(spmse_df):
    """
    Test that colorbar labels match the bin_labels.
    """

    fig = plot_spmse(spmse_df, None, False)
    colorbar = fig.data[0].colorbar

    output = [
        "CONSTANT VARIABLE",
        "(0,3]",
        "(3,10]",
        "(10,30]",
        "(30,100]",
        "(100,+)",
    ]

    assert list(colorbar.ticktext) == output


def test_colourscale(spmse_df):
    """
    Test that colorbar labels match the bin_labels.
    """

    fig = plot_spmse(spmse_df, None, False)
    colorscale = fig.data[0].colorscale

    output = (
        (0.0, 'rgb(255,255,255)'),
        (1/6, 'rgb(255,255,255)'),
        (1/6, 'rgb(255,255,229)'),
        (2/6, 'rgb(255,255,229)'),
        (2/6, 'rgb(255,247,188)'),
        (0.5, 'rgb(255,247,188)'),
        (0.5, 'rgb(254,227,145)'),
        (4/6, 'rgb(254,227,145)'),
        (4/6, 'rgb(254,196,79)'),
        (5/6, 'rgb(254,196,79)'),
        (5/6, 'rgb(254,153,41)'),
        (1.0, 'rgb(254,153,41)')
    )

    assert (output == colorscale)


def test_layout_properties(spmse_df):
    """
    Test that figure layout has correct title, width, height.
    """

    fig = plot_spmse(spmse_df, None, False)

    assert fig.layout.title.text == "S_pMSE Heatmap"
    assert fig.layout.width == 986
    assert fig.layout.height == 850


def test_hover_template(spmse_df):
    """
    Test that checks whether the correct hovertemplate is used
    """

    fig = plot_spmse(spmse_df, None, False)

    assert fig.data[0].hovertemplate == (
        "x: %{x}<br>"
        "y: %{y}<br>"
        "value: %{text}"
        "<extra></extra>"
    )


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

    outfile = tmp_path / "spmse.pdf"
    plot_spmse(spmse_df, str(outfile), False)

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
    """Test if function does not change the input"""

    original_df = spmse_df.copy(deep=True)
    plot_spmse(spmse_df, None, False)

    pd.testing.assert_frame_equal(spmse_df, original_df)
