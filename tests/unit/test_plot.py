import tempfile
import webbrowser
from pathlib import Path

import pytest
import pandas as pd

from synthpop.plotting.plot import plot_univariate_distributions

# ----- univariate distribution tests -----
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
            saving_location=None
        )

def test_syn_df_must_be_dataframe():
    with pytest.raises(ValueError, match="synthetic data should be a pandas DataFrame"):
        plot_univariate_distributions(
            orig_df=pd.DataFrame({"x": [1, 2]}),
            syn_df=[],
            saving_location=None
        )

def test_column_mismatch_raises():
    orig = pd.DataFrame({"a": [1, 2]})
    syn = pd.DataFrame({"b": [1, 2]})

    with pytest.raises(ValueError, match="datasets must have identical columns"):
        plot_univariate_distributions(orig, syn, None)

def test_no_save_and_no_browser_when_non_interactive(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        saving_location=None,
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
        saving_location="/some/folder",
        interactive=False
    )

    assert len(mocked_environment["mkdir_calls"]) == 1
    assert len(mocked_environment["write_calls"]) == 1
    assert len(mocked_environment["tempfile_calls"]) == 0
    assert len(mocked_environment["browser_calls"]) == 0

def test_save_and_browser_when_interactive(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        saving_location="/some/folder",
        interactive=True,
    )

    assert len(mocked_environment["mkdir_calls"]) == 1
    assert len(mocked_environment["write_calls"]) == 1
    assert len(mocked_environment["tempfile_calls"]) == 0
    assert len(mocked_environment["browser_calls"]) == 1
    
def test_browser_opens_when_interactive_without_save_location(mocked_environment):
    orig = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        saving_location=None,
        interactive=True,
    )

    assert len(mocked_environment["mkdir_calls"]) == 0
    assert len(mocked_environment["write_calls"]) == 0
    assert len(mocked_environment["tempfile_calls"]) == 1
    assert mocked_environment["tempfile_html"] is not None
    assert len(mocked_environment["browser_calls"]) == 1

def test_written_html_contains_column_name(mocked_environment):
    orig = pd.DataFrame({"age": [20, 30, 40]})
    syn = pd.DataFrame({"age": [21, 31, 41]})

    plot_univariate_distributions(
        orig,
        syn,
        saving_location="/some/folder",
    )

    html = mocked_environment["written_html"]

    assert "age" in html
    assert "Distribution comparison: age" in html

def test_missing_value_annotation_in_html(mocked_environment):
    orig = pd.DataFrame({"x": [1, None, 3]})
    syn = pd.DataFrame({"x": [None, 2, 3]})

    plot_univariate_distributions(
        orig,
        syn,
        saving_location="/some/folder",
    )

    html = mocked_environment["written_html"]

    assert "Original: 1" in html
    assert "Synthetic: 1" in html
