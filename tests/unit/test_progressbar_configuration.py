import time

import pytest
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from synthpop.utils import _progress_bar


def test_progress_bar_is_Progressclass():
    progress = _progress_bar()
    assert isinstance(progress, Progress)


def test_progress_bar_spinner_column():
    progress = _progress_bar()

    column = progress.columns[0]
    assert isinstance(column, SpinnerColumn)
    assert column.spinner.name == "line"
    assert column.spinner.style == "yellow"


def test_progress_bar_text_column():
    progress = _progress_bar()

    column = progress.columns[1]
    assert isinstance(column, TextColumn)
    assert column.text_format == "{task.description}"


def test_progress_bar_bar_column():
    progress = _progress_bar()

    column = progress.columns[2]
    assert isinstance(column, BarColumn)
    assert column.complete_style == "dark_orange"
    assert column.finished_style == "bold green"
    assert column.pulse_style == "dark_orange"


def test_progress_bar_non_configurableparams():
    progress = _progress_bar()

    assert len(progress.columns) == 6
    assert isinstance(progress.columns[3], MofNCompleteColumn)
    assert isinstance(progress.columns[4], TimeElapsedColumn)
    assert isinstance(progress.columns[5], TimeRemainingColumn)


def test_progress_bar_transient():
    progress = _progress_bar(transient=True)

    assert progress.live.transient
    assert not progress.disable


def test_progress_bar_disabled():
    progress = _progress_bar(disable=True)

    assert not progress.live.transient
    assert progress.disable


def test_progress_bar_transient_and_disabled():
    progress = _progress_bar(transient=True, disable=True)

    assert progress.live.transient
    assert progress.disable


# def test_visual():
#     """
#     run in terminal: pytest -s tests/unit/test_progressbar_configuration.py::test_visual
#     """
#     progressbar = _progress_bar()
#     columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     with progressbar as progress:
#         task = progress.add_task(
#             "[bold magenta]Fitting[/]",
#             total=len(columns),
#         )

#         for item in columns:
#             progress.update(
#                 task, advance=1,
#                 description=f"[bold magenta]Fitting[/] - [cyan]column:[/] {item}",
#             )

#             time.sleep(1.)
