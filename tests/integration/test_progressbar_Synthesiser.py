import numpy as np
import pandas as pd
import pytest
from rich.progress import Progress, TextColumn

from synthpop import Synthesiser


def mock_bar(disable = False, transient = False) -> Progress:
    return Progress(TextColumn('test bar'))


def test_synthesiser_progressbar_update_and_add_task_is_called_during_fit(mocker, monkeypatch):
    mockerpbar = mocker.patch(
            "synthpop.synthesiser._progress_bar",
            wraps = mock_bar,)
    
    X = pd.DataFrame(
        {
            'a': [1, 2, 3, 4],
            'b': ['this', 'is', 'a', 'string'],
            'c': ['this', 'too', 'but also with',  np.nan],
            'd': [1.,2.,3.,4.]
        }
    )

    called = False

    def fake_add_task(*args, **kwargs):
        nonlocal called
        called = True

    num =0

    def fake_add_update(*args, **kwargs):
        nonlocal num
        num +=1

    monkeypatch.setattr(Progress, "add_task", fake_add_task)
    monkeypatch.setattr(Progress, "update", fake_add_update)

    Synthesiser(random_seed=0).fit(X)

    mockerpbar.assert_called_once()
    assert called
    assert num == 4 

def test_synthesiser_progressbar_verbose_is_True(mocker):
    mockerpbar = mocker.patch(
            "synthpop.synthesiser._progress_bar",
            wraps = mock_bar,)
    
    X = pd.DataFrame(
        {
            'a': [1, 2, 3, 4],
            'b': ['this', 'is', 'a', 'string'],
            'c': ['this', 'too', 'but also with',  np.nan],
            'd': [1.,2.,3.,4.]
        }
    )

    Synthesiser(random_seed=0, verbose=1).fit(X)
    
    args, kwargs = mockerpbar.call_args
    mockerpbar.assert_called_once()
    assert kwargs == {'disable': False, 'transient': False}

def test_synthesiser_progressbar_verbose_is_False(mocker):
    mockerpbar = mocker.patch(
            "synthpop.synthesiser._progress_bar",
            wraps = mock_bar,)
    
    X = pd.DataFrame(
        {
            'a': [1, 2, 3, 4],
            'b': ['this', 'is', 'a', 'string'],
            'c': ['this', 'too', 'but also with',  np.nan],
            'd': [1.,2.,3.,4.]
        }
    )

    Synthesiser(random_seed=0, verbose=0).fit(X)
    
    args, kwargs = mockerpbar.call_args

    mockerpbar.assert_called_once()
    assert kwargs == {'disable': True, 'transient': True}