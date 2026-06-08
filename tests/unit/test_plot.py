import pytest

import pandas as pd
from pathlib import Path

from synthpop.plotting.plot import plot_univariate_distributions

# ----- univariate distribution tests -----
@pytest.fixture
def mocked_filesystem(monkeypatch):
    state = {
        "mkdir_calls": [],
        "write_calls": [],
        "written_html": None,
    }

    def fake_mkdir(*args, **kwargs):
        state["mkdir_calls"].append((args, kwargs))

    def fake_write_text(self, text, *args, **kwargs):
        state["write_calls"].append((self, text))
        state["written_html"] = text

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    return state

def test_obs_df_must_be_dataframe():
    with pytest.raises(ValueError, match="observed data should be a pandas DataFrame"):
        plot_univariate_distributions(
            obs_df=[],
            syn_df=pd.DataFrame({"x": [1, 2]}),
            saving_location=None
        )

def test_syn_df_must_be_dataframe():
    with pytest.raises(ValueError, match="synthetic data should be a pandas DataFrame"):
        plot_univariate_distributions(
            obs_df=pd.DataFrame({"x": [1, 2]}),
            syn_df=[],
            saving_location=None
        )

def test_column_mismatch_raises():
    obs = pd.DataFrame({"a": [1, 2]})
    syn = pd.DataFrame({"b": [1, 2]})

    with pytest.raises(ValueError, match="datasets must have identical columns"):
        plot_univariate_distributions(obs, syn, None)

def test_no_save_when_location_none(mocked_filesystem):
    obs = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        obs,
        syn,
        saving_location=None,
    )

    assert mocked_filesystem["mkdir_calls"] == []
    assert len(mocked_filesystem["write_calls"]) == 0

def test_save_called_when_location_provided(mocked_filesystem):
    obs = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})

    plot_univariate_distributions(
        obs,
        syn,
        saving_location="/some/folder",
    )

    assert len(mocked_filesystem["mkdir_calls"]) == 1
    assert len(mocked_filesystem["write_calls"]) == 1

def test_written_html_contains_column_name(mocked_filesystem):
    obs = pd.DataFrame({"age": [20, 30, 40]})
    syn = pd.DataFrame({"age": [21, 31, 41]})

    plot_univariate_distributions(
        obs,
        syn,
        saving_location="/some/folder",
    )

    html = mocked_filesystem["written_html"]

    assert "age" in html
    assert "Distribution comparison: age" in html

def test_missing_value_annotation_in_html(mocked_filesystem):
    obs = pd.DataFrame({"x": [1, None, 3]})
    syn = pd.DataFrame({"x": [None, 2, 3]})

    plot_univariate_distributions(
        obs,
        syn,
        saving_location="/some/folder",
    )

    html = mocked_filesystem["written_html"]

    assert "Observed: 1" in html
    assert "Synthetic: 1" in html
