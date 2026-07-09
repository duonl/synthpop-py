import tempfile
import webbrowser
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from synthpop.plotting.plot_univariate import (
    plot_univariate_distributions,
    _make_histograms,
    _make_bars,
    _plot_single_distribution,
    _build_html,
    _write_html,
)


# ----- _make_histograms tests -----

def test_make_histograms_returns_histograms():
    orig = pd.Series([1, 2, 3, pd.NA])
    syn = pd.Series([10, 11, 12, pd.NA])

    orig_hist, syn_hist = _make_histograms(orig, syn)

    assert isinstance(orig_hist, go.Histogram)
    assert orig_hist.x.tolist() == orig.dropna().tolist()
    assert orig_hist.name == "Original"

    assert isinstance(syn_hist, go.Histogram)
    assert syn_hist.x.tolist() == syn.dropna().tolist()
    assert syn_hist.name == "Synthetic"

def test_make_histograms_corrects_bins_for_integer_data():
    orig = pd.Series([1, 2, 3])
    syn = pd.Series([4, 5, 6])

    orig_hist, syn_hist = _make_histograms(orig, syn)

    # shared binning logic
    for histogram in [orig_hist, syn_hist]:
        assert histogram.xbins.start == 0.5
        assert histogram.xbins.end == 6.5
        assert histogram.xbins.size == 1
        assert histogram.histnorm == "probability density"

def test_make_histograms_float_bin_logic(monkeypatch):
    orig = pd.Series([1.1, 2.2, 3.3])
    syn = pd.Series([4.4, 5.5, 6.6])

    fake_bins = np.array([0.0, 10.0, 20.0])

    called = {"args": None}

    def fake_histogram_bin_edges(arr, bins):
        called["args"] = (arr, bins)
        return fake_bins
    
    monkeypatch.setattr(
        np,
        "histogram_bin_edges",
        fake_histogram_bin_edges,
    )

    orig_hist, syn_hist = _make_histograms(orig, syn)

    assert called["args"][1] == "auto"

    expected_bin_size = float(fake_bins[1] - fake_bins[0])

    for histogram in [orig_hist, syn_hist]:
        assert isinstance(histogram, go.Histogram)
        assert histogram.xbins.start == fake_bins[0]
        assert histogram.xbins.end == fake_bins[-1]
        assert histogram.xbins.size == expected_bin_size
        assert histogram.histnorm == "probability density"
    
    assert syn_hist.xbins == orig_hist.xbins

@pytest.mark.parametrize(
    "orig, syn",
    [
        (pd.Series([1, None, 2]), pd.Series([3, 4, pd.NA])),
        (pd.Series([1.0, np.nan, 2.0]), pd.Series([3.0, 4.0, np.nan])),
    ],
)
def test_make_histograms_removes_missing_data(orig, syn):
    orig_hist, syn_hist = _make_histograms(orig, syn)

    all(not pd.isna(x) for x in orig_hist.x)
    all(not pd.isna(x) for x in syn_hist.x)


# ----- _make_bars tests -----

def test_make_bars_aligns_categories_and_normalises():
    orig = pd.Series(["A", "A", "B"])
    syn = pd.Series(["A", "B", "B", "B"])

    orig_bar, syn_bar = _make_bars(orig, syn)

    assert list(orig_bar.x) == ["A", "B"]
    assert list(syn_bar.x) == ["A", "B"]

    assert pytest.approx(orig_bar.y) == [2/3, 1/3]
    assert pytest.approx(syn_bar.y) == [1/4, 3/4]

def test_make_bars_handles_missing_values():
    orig = pd.Series(["A", pd.NA, "B"])
    syn = pd.Series([None, "A", "A"])

    orig_bar, syn_bar = _make_bars(orig, syn)

    assert "<MISSING>" in orig_bar.x
    assert "<MISSING>" in syn_bar.x

    # missing counts are encoded in customdata
    assert sum(orig_bar.customdata) == 3
    assert sum(syn_bar.customdata) == 3

def test_make_bars_does_not_fail_with_categorical_dtype_and_missing_values():
    """
    Regression test for:
    TypeError: Cannot setitem on a Categorical with a new category (<MISSING>)
    """

    orig = pd.Series(["A", "B", None, "A"], dtype="category")
    syn = pd.Series(["A", None, "B", "B"], dtype="category")

    # Should NOT raise
    orig_bar, syn_bar = _make_bars(orig, syn)

    for bar in [orig_bar, syn_bar]:
        assert isinstance(bar, go.Bar)  # basic sanity check: output must be go.Bar
        assert '<MISSING>' in bar.x # ensure missing handling worked
        assert pytest.approx(sum(bar.y)) == 1.0 # basic sanity check: densities should sum to 1
        assert sum(bar.customdata) == 4 # should match row count including missing

# ----- _plot_single_distribution tests -----

def test_plot_single_distribution_uses_histograms(monkeypatch):
    orig = pd.Series([1, 2, 3])
    syn = pd.Series([4, 5, 6])

    called = {"hist": 0, "bars": 0, "hist_args": None, "bar_args": None}

    def fake_hist(orig, syn):
        called["hist_args"] = (orig, syn)
        called["hist"] += 1
        return go.Histogram(), go.Histogram()

    def fake_bars(orig, syn):
        called["bar_args"] = (orig, syn)
        called["bars"] += 1
        return go.Bar(), go.Bar()

    monkeypatch.setattr("synthpop.plotting.plot_univariate._make_histograms", fake_hist)
    monkeypatch.setattr("synthpop.plotting.plot_univariate._make_bars", fake_bars)

    fig = _plot_single_distribution(orig, syn, "age")

    assert called["hist"] == 1
    assert called["bars"] == 0
    assert len(fig.data) == 2

    assert called["hist_args"][0].equals(orig)
    assert called["hist_args"][1].equals(syn)

def test_plot_single_distribution_categorical_path(monkeypatch):
    orig = pd.Series(["A", "B"])
    syn = pd.Series(["A", "A"])

    called = {"hist": 0, "bars": 0, "hist_args": None, "bar_args": None}

    def fake_hist(orig, syn):
        called["hist_args"] = (orig, syn)
        called["hist"] += 1
        return go.Histogram(), go.Histogram()

    def fake_bars(orig, syn):
        called["bar_args"] = (orig, syn)
        called["bars"] += 1
        return go.Bar(), go.Bar()

    monkeypatch.setattr("synthpop.plotting.plot_univariate._make_histograms", fake_hist)
    monkeypatch.setattr("synthpop.plotting.plot_univariate._make_bars", fake_bars)

    fig = _plot_single_distribution(orig, syn, "sex")

    assert called["bars"] == 1
    assert called["hist"] == 0
    assert len(fig.data) == 2

    assert called["bar_args"][0].equals(orig)
    assert called["bar_args"][1].equals(syn)

def test_plot_single_distribution_adds_annotation():
    orig = pd.Series([1, None, 2])
    syn = pd.Series([None, 2, 3])

    fig = _plot_single_distribution(orig, syn, "age")

    annotation = fig.layout.annotations[0].text

    assert annotation == (
        "Missing values - Original: 1, Synthetic: 1"
    )

def test_plot_single_distribution_uses_column_name_in_title():
    orig = pd.Series([1, 2, 3])
    syn = pd.Series([4, 5, 6])

    fig = _plot_single_distribution(orig, syn, "age")

    assert fig.layout.title.text == "Distribution comparison: age"

def test_plot_single_distribution_numeric_layout():
    orig = pd.Series([1, 2, 3])
    syn = pd.Series([4, 5, 6])

    fig = _plot_single_distribution(orig, syn, "age")

    assert fig.layout.barmode == "overlay"
    assert fig.layout.xaxis.title.text == "age"
    assert fig.layout.yaxis.title.text == "Density"
    assert fig.layout.legend.title.text == "Dataset"
    assert fig.layout.height == 500

def test_plot_single_distribution_categorical_layout():
    orig = pd.Series(["A", "B"])
    syn = pd.Series(["A", "A"])

    fig = _plot_single_distribution(orig, syn, "sex")

    assert fig.layout.barmode == "group"
    assert fig.layout.xaxis.title.text == "sex"
    assert fig.layout.yaxis.title.text == "Density"
    assert fig.layout.legend.title.text == "Dataset"
    assert fig.layout.height == 500

# ----- _build_html tests -----

def test_build_html_uses_correct_to_html_arguments(monkeypatch):
    figs = [go.Figure(), go.Figure(), go.Figure()]

    calls = []

    def fake_to_html(fig, include_plotlyjs, full_html):
        calls.append(
            {
                "include_plotlyjs": include_plotlyjs,
                "full_html": full_html,
            }
        )
        return "<div>FIG</div>"

    monkeypatch.setattr(
        "synthpop.plotting.plot_univariate.to_html",
        fake_to_html,
    )

    _build_html(figs)

    assert calls == [
        {
            "include_plotlyjs": True,
            "full_html": False,
        },
        {
            "include_plotlyjs": False,
            "full_html": False,
        },
        {
            "include_plotlyjs": False,
            "full_html": False,
        },
    ]

# ----- _write_html tests -----

def test_write_html_uses_named_tempfile(monkeypatch):
    captured = {}

    class FakeTemp:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.name = "/tmp/fake.html"
            self.written = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, content):
            self.written = content

    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        lambda *a, **k: FakeTemp(*a, **k),
    )

    html = "<html></html>"
    path = _write_html(html, None)

    assert captured["kwargs"]["mode"] == "w"
    assert captured["kwargs"]["suffix"] == ".html"
    assert captured["kwargs"]["delete"] is False
    assert captured["kwargs"]["encoding"] == "utf-8"

    assert str(path).endswith(".html")


# ----- plot_univariate_distribution tests -----

@pytest.fixture
def mocked_environment(monkeypatch):
    state = {
        "mkdir_calls": [],
        "write_calls": [],
        "written_html": None,
        "browser_calls": [],
        "tempfile_calls": [],
        "tempfile_html": None,
    }

    def fake_mkdir(*args, **kwargs):
        state["mkdir_calls"].append((args, kwargs))

    def fake_write_text(self, text, *args, **kwargs):
        state["write_calls"].append((self, text))
        state["written_html"] = text

    def fake_browser_open(*args, **kwargs):
        state["browser_calls"].append((args, kwargs))

    class FakeTempFile:
        name = "/fake/temp/univariate_distribution_comparison.html"

        def write(self, text):
            state["tempfile_html"] = text

        def __enter__(self):
            state["tempfile_calls"].append(True)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(Path, "write_text", fake_write_text)
    monkeypatch.setattr(webbrowser, "open", fake_browser_open)
    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: FakeTempFile(),
    )

    return state

def test_orig_df_must_be_dataframe():
    with pytest.raises(ValueError, match="original data should be a pandas DataFrame"):
        plot_univariate_distributions(
            orig_df=[],
            syn_df=pd.DataFrame({"x": [1, 2]}),
            save_path=None
        )

def test_syn_df_must_be_dataframe():
    with pytest.raises(ValueError, match="synthetic data should be a pandas DataFrame"):
        plot_univariate_distributions(
            orig_df=pd.DataFrame({"x": [1, 2]}),
            syn_df=[],
            save_path=None
        )

def test_column_mismatch_raises():
    orig = pd.DataFrame({"a": [1, 2]})
    syn = pd.DataFrame({"b": [1, 2]})

    with pytest.raises(ValueError, match="datasets must have identical columns"):
        plot_univariate_distributions(orig, syn, None)
    
def test_output_is_go_figure():
    orig = pd.DataFrame({"a": [1, 2], "b": ["a", "b"]})
    syn = pd.DataFrame({"a": [1, 2], "b": ["a", "b"]})

    result = plot_univariate_distributions(orig, syn, None, False)

    assert isinstance(result, list)
    assert len(result) == len(orig.columns)
    assert all(isinstance(fig, go.Figure) for fig in result)

def test_no_save_and_no_browser_when_non_interactive(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        save_path=None,
        interactive=False
    )

    assert len(mocked_environment["mkdir_calls"]) == 0
    assert len(mocked_environment["write_calls"]) == 0
    assert len(mocked_environment["tempfile_calls"]) == 0
    assert len(mocked_environment["browser_calls"]) == 0

def test_save_and_no_browser_when_non_interactive(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        save_path="/some/folder",
        interactive=False
    )

    assert len(mocked_environment["mkdir_calls"]) == 1
    _, mkdir_kwargs = mocked_environment["mkdir_calls"][0]
    assert mkdir_kwargs["parents"] is True
    assert mkdir_kwargs["exist_ok"] is True

    assert len(mocked_environment["write_calls"]) == 1
    path_obj, _ = mocked_environment["write_calls"][0]
    assert path_obj.name == "univariate_distribution_comparison.html"

    assert len(mocked_environment["tempfile_calls"]) == 0
    assert len(mocked_environment["browser_calls"]) == 0

def test_save_and_browser_when_interactive(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        save_path="/some/folder",
        interactive=True,
    )

    assert len(mocked_environment["mkdir_calls"]) == 1
    _, mkdir_kwargs = mocked_environment["mkdir_calls"][0]
    assert mkdir_kwargs["parents"] is True
    assert mkdir_kwargs["exist_ok"] is True

    assert len(mocked_environment["write_calls"]) == 1
    path_obj, _ = mocked_environment["write_calls"][0]
    assert path_obj.name == "univariate_distribution_comparison.html"

    assert len(mocked_environment["tempfile_calls"]) == 0

    assert len(mocked_environment["browser_calls"]) == 1    
    browser_args, _ = mocked_environment["browser_calls"][0]
    assert browser_args[0].endswith("univariate_distribution_comparison.html")
    
def test_browser_opens_when_interactive_without_save_location(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        save_path=None,
        interactive=True,
    )

    assert len(mocked_environment["mkdir_calls"]) == 0
    assert len(mocked_environment["write_calls"]) == 0
    assert len(mocked_environment["tempfile_calls"]) == 1
    assert mocked_environment["tempfile_html"] is not None
    
    assert len(mocked_environment["browser_calls"]) == 1
    browser_args, _ = mocked_environment["browser_calls"][0]
    assert browser_args[0] == (
        Path("/fake/temp/univariate_distribution_comparison.html")
        .resolve()
        .as_uri()
    )

def test_plot_univariate_distributions_flow(monkeypatch):
    orig = pd.DataFrame({"a": [1, 2], "b": ["1", "2"]})
    syn = pd.DataFrame({"a": [3, 4], "b": ["1", "2"]})

    fake_fig = go.Figure()
    fake_html = "<html>HTML</html>"
    fake_path = Path("/tmp/final.html")

    captured = {
        "plots_input": None,
        "html_input": None,
        "write_input": None,
        "browser_input": None,
    }

    def fake_plot(orig, syn, name):
        return fake_fig

    def fake_build_html(figs):
        captured["plots_input"] = figs
        return fake_html

    def fake_write(html, save_path):
        captured["html_input"] = html
        return fake_path

    def fake_browser(url):
        captured["browser_input"] = url

    monkeypatch.setattr(
        "synthpop.plotting.plot_univariate._plot_single_distribution",
        fake_plot,
    )
    monkeypatch.setattr(
        "synthpop.plotting.plot_univariate._build_html",
        fake_build_html,
    )
    monkeypatch.setattr(
        "synthpop.plotting.plot_univariate._write_html",
        fake_write,
    )
    monkeypatch.setattr(
        "synthpop.plotting.plot_univariate.webbrowser.open",
        fake_browser,
    )

    result = plot_univariate_distributions(
        orig,
        syn,
        save_path="x",
        interactive=True,
    )

    assert captured["plots_input"] == [fake_fig, fake_fig]
    assert captured["html_input"] == fake_html
    assert captured["browser_input"] == fake_path.resolve().as_uri()
    assert result == [fake_fig, fake_fig]