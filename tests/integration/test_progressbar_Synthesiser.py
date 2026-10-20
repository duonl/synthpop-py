import numpy as np
import pandas as pd
import pytest
from rich.progress import Progress, TextColumn

from synthpop import Synthesiser


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["this", "is", "a", "string"],
        "c": ["this", "too", "but also with", np.nan],
        "d": [1., 2., 3., 4.],
    })


def mock_bar(disable=False, transient=False) -> Progress:
    return Progress(TextColumn('test bar'))


def test_synthesiser_progressbar_is_called_during_fit(
    mocker, sample_data
):
    mockerpbar = mocker.patch(
        "synthpop.synthesiser._progress_bar",
        wraps=mock_bar,
    )

    Synthesiser(random_seed=0, verbose=1).fit(sample_data)

    mockerpbar.assert_called_once()

def test_synthesiser_progressbar_adds_task_during_fit(
    mocker, sample_data
):
    mocker.patch(
        "synthpop.synthesiser._progress_bar",
        wraps=mock_bar,
    )
    add_task = mocker.patch.object(
        Progress, "add_task")

    # As else an error is raised by Rich. Update itself will be tested in a different test.
    mocker.patch.object(Progress, "update")
    Synthesiser(random_seed=0, verbose=1).fit(sample_data)

    add_task.assert_called_once_with("Fitting", total=4)


def test_synthesiser_progressbar_updates_each_column_during_fit(
    mocker, sample_data
):
    mocker.patch(
        "synthpop.synthesiser._progress_bar",
        wraps=mock_bar,
    )
    mocker.patch.object(
        Progress, "add_task", return_value=123
    )
    update = mocker.patch.object(Progress, "update")

    Synthesiser(random_seed=0, verbose=1).fit(sample_data)

    expected_descriptions = [
        "[bold magenta]Fitting[/] - [cyan]column:[/] a",
        "[bold magenta]Fitting[/] - [cyan]column:[/] b",
        "[bold magenta]Fitting[/] - [cyan]column:[/] c",
        "[bold magenta]Fitting[/] - [cyan]column:[/] d",
    ]

    assert update.call_count == 4

    assert [
        (args, kwargs)
        for args, kwargs in update.call_args_list
    ] == [
        (
            (123,),
            {"advance": 1, "description": description},
        )
        for description in expected_descriptions
    ]


@pytest.mark.parametrize(
    ("verbose", "expected_kwargs"),
    [
        (1, {"disable": False, "transient": False}),
        (0, {"disable": True, "transient": True}),
    ],
)
def test_synthesiser_progressbar_verbose(
    mocker, sample_data, verbose, expected_kwargs
):
    mockerpbar = mocker.patch(
        "synthpop.synthesiser._progress_bar",
        wraps=mock_bar,
    )

    Synthesiser(
        random_seed=0, verbose=verbose
    ).fit(sample_data)

    mockerpbar.assert_called_once_with(**expected_kwargs)
