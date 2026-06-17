import numpy as np
import pytest
import pandas as pd

from synthpop.plotting.plot_spmse import plot_spmse


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


@pytest.mark.parametrize(
    "df",
    [
        (
            pd.DataFrame(
                {
                    "column1": ["c1"],
                    "column2": [2],
                    "column3": ["c3"],
                    "column4": [2]
                }
            )
        ),  # Can only be exactly 3 columns

        pd.DataFrame(
            {
                "column1": ["c1"],
                "column2": ["c1"],
                "spmse": [1.0]
            }
        )  # Wrong capitalization
    ],
)
def test_input_errors(df):
    """
    Test that invalid column names raise a ValueError.
    """

    with pytest.raises(ValueError, match="should be of shape 3xN"):
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
    Test that cells with 0 S_pMSE are shown as 'MISSING'.
    """

    fig = plot_spmse(spmse_df, None, False)
    text = np.array(fig.data[0].text)

    output = np.array(
        [
            ['12.46', '46.49', '0.0'],
            ['473842.49', '4.0', '46.49'],
            ['MISSING', '473842.49', '12.46']
        ]
    )

    assert (output == text).all()


def test_colorbar_labels(spmse_df):
    """
    Test that colorbar labels match the bin_labels.
    """

    fig = plot_spmse(spmse_df, None, False)
    colorbar = fig.data[0].colorbar

    output = [
        "MISSING",
        "(0,3]",
        "(3,10]",
        "(10,30]",
        "(30,100]",
        "(100,+)",
    ]

    assert list(colorbar.ticktext) == output


def test_colorscale(spmse_df):
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
    assert fig.layout.width == 900
    assert fig.layout.height == 845


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

    outfile = tmp_path / "plot.png"
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

def test_no_input_change(spmse_df):
    """Test if function does not change the input"""

    original_df = spmse_df.copy(deep=True)
    plot_spmse(spmse_df, None, False)

    pd.testing.assert_frame_equal(spmse_df, original_df)