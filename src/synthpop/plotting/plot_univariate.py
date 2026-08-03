"""
This module contains a function to visually inspect the univariate distributions. 
"""
from pathlib import Path
import tempfile
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

original_colour = "#004488"
synthetic_colour = "#DDAA33"

def _make_histograms(orig: pd.Series, syn: pd.Series) -> tuple[go.Histogram, go.Histogram]:
    
    orig_non_missing = orig.dropna()
    syn_non_missing = syn.dropna()

    combined = pd.concat(
        [orig_non_missing, syn_non_missing],
        ignore_index=True,
    )

    if len(combined) == 0:
        return(
            go.Histogram(x=[], name="Original"),
            go.Histogram(x=[], name="Synthetic")
        )
    
    is_integer = (
        pd.api.types.is_integer_dtype(orig)
        and pd.api.types.is_integer_dtype(syn)
    )

    if is_integer:
        min_val = int(combined.min())
        max_val = int(combined.max())

        xbins = dict(
            start=min_val - 0.5,
            end=max_val + 0.5,
            size=1,
        )

    else:
        bins = np.histogram_bin_edges(
            combined.to_numpy(),
            bins="auto",
        )

        xbins = dict(
            start=float(bins[0]),
            end=float(bins[-1]),
            size=float(bins[1] - bins[0]),
        )

    orig_hist = go.Histogram(
        x=orig_non_missing,
        name="Original",
        histnorm="probability density",
        xbins=xbins,
        opacity=0.6,
        marker_color = original_colour,
    )

    syn_hist = go.Histogram(
        x=syn_non_missing,
        name="Synthetic",
        histnorm="probability density",
        xbins=xbins,
        opacity=0.6,
        marker_color = synthetic_colour,
    )

    return orig_hist, syn_hist

def _make_bars(orig: pd.Series, syn: pd.Series) -> tuple[go.Bar, go.Bar]:

    # set input Series to 'string' to avoid 
    # "TypeError: Cannot setitem on a Categorical with a new category (<MISSING>)"
    # when the input is category dtype
    orig_str = orig.astype("string").fillna("<MISSING>")
    syn_str = syn.astype("string").fillna("<MISSING>")

    orig_counts = orig_str.value_counts(normalize=True, dropna=False)
    syn_counts = syn_str.value_counts(normalize=True, dropna=False)

    levels = sorted(
        set(orig_counts.index.astype(str))
        | set(syn_counts.index.astype(str))
    )

    orig_density = [
        float(orig_counts.get(level, 0.0))
        for level in levels
    ]

    syn_density = [
        float(syn_counts.get(level, 0.0))
        for level in levels
    ]

    orig_bar = go.Bar(
        x=levels,
        y=orig_density,
        name="Original",
        marker_color = original_colour,
        customdata=[
            int(
                (orig_str.fillna("<MISSING>") == level).sum()
            )
            for level in levels
        ],
        hovertemplate=(
            "Level=%{x}<br>"
            "Density=%{y:.4f}<br>"
            "Count=%{customdata}"
            "<extra>Original</extra>"
        ),
    )

    syn_bar = go.Bar(
        x=levels,
        y=syn_density,
        name="Synthetic",
        marker_color = synthetic_colour,
        customdata=[
            int(
                (syn_str.fillna("<MISSING>") == level).sum()
            )
            for level in levels
        ],
        hovertemplate=(
            "Level=%{x}<br>"
            "Density=%{y:.4f}<br>"
            "Count=%{customdata}"
            "<extra>Synthetic</extra>"
        ),
    )

    return orig_bar, syn_bar

def _plot_single_distribution(orig: pd.Series, syn: pd.Series, column: str) -> go.Figure:

    fig = go.Figure()

    orig_missing = int(orig.isna().sum())
    syn_missing = int(syn.isna().sum())

    if pd.api.types.is_numeric_dtype(orig) and not pd.api.types.is_bool_dtype(orig):

        orig_hist, syn_hist = _make_histograms(orig, syn)

        fig.add_trace(orig_hist)
        fig.add_trace(syn_hist)

        fig.update_layout(
            title=f"Distribution comparison: {column}",
            xaxis_title=column,
            yaxis_title="Density",
            barmode="overlay",
        )

    else:

        orig_bar, syn_bar = _make_bars(orig, syn)

        fig.add_trace(orig_bar)
        fig.add_trace(syn_bar)

        fig.update_layout(
            title=f"Distribution comparison: {column}",
            xaxis_title=column,
            yaxis_title="Density",
            barmode="group",
        )

    fig.add_annotation(
        text=(
            f"Missing values - "
            f"Original: {orig_missing}, "
            f"Synthetic: {syn_missing}"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.20,
        showarrow=False,
        align="left",
    )

    fig.update_layout(
        legend_title_text="Dataset",
        height=500,
        margin=dict(b=100),
    )

    return fig

def _build_html(figures: list[go.Figure]) -> str:

    html_parts = [
        """
        <html>
        <head>
            <meta charset="utf-8">
            <title>Univariate Distribution Comparison</title>
        </head>
        <body>
        <h1>Univariate Distribution Comparison</h1>
        """
    ]

    for idx, fig in enumerate(figures):
        html_parts.append(
            to_html(
                fig,
                include_plotlyjs=(idx == 0),
                full_html=False,
            )
        )

    html_parts.append("</body></html>")

    return "\n".join(html_parts)

def _write_html(html_content: str, save_path: str | None) -> Path:

    if save_path is None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write(html_content)

        return Path(temp_file.name)

    output_dir = Path(save_path)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "univariate_distribution_comparison.html"
    )

    output_file.write_text(
        html_content,
        encoding="utf-8",
    )

    return output_file


def plot_univariate_distributions(
        orig_df: pd.DataFrame, 
        syn_df: pd.DataFrame, 
        save_path: str | None =  None,
        interactive: bool = False,
        ) -> list[go.Figure]:
    """
    Create interactive univariate distribution plots comparing original and 
    synthetic datasets.

    For each variable in the datasets, a separate Plotly visualisation is
    generated and added to a single HTML document. Numeric variables are
    displayed as overlapping density histograms, while categorical and
    boolean variables are displayed as side-by-side relative frequency bar
    charts. Missing value counts for both datasets are included as plot
    annotations.

    The original and synthetic datasets must contain identical columns.
    Variables are processed independently and visualised sequentially in a 
    single HTML document to support scrolling and browser-based search.

    If a saving location is provided, the generated HTML document is written
    to `univariate_distribution_comparison.html` in the specified directory.
    If interactive mode is enabled, the resulting HTML document is
    automatically opened in the default web browser. When no saving location
    is provided, a temporary HTML file is created and opened only when
    interactive mode is enabled.
    
    :param orig_df: Original/observed dataset.
    :param syn_df: Synthetic dataset. Must contain the same columns as `orig_df`.
    :param save_path: Directory where the HTML output file will be written.
        If a relative path is provided, it is resolved relative to the current working
        directory. The directory is created if it does not already exist (including)
        parent directories). If `None` (default), no permanent output file is created.
    :param interactive: Whether to automatically open the generated visualisation
        in the default web browser. Default is `False`. When running headless,
        the parameter should be set to `False`.

    :return: List of plots.

    Notes
    -----
    Histograms are normalised to probability densities to allow comparison
    between datasets of different sizes. For categorical and boolean variables,
    relative frequencies are displayed and category levels missing from either dataset
    are assigned a density of zero to maintain comparability.

    Examples
    --------
    Using the default parameters:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from synthpop.plotting import plot_univariate_distributions
    >>>
    >>> np.random.seed(42)
    >>>
    >>> orig_df = pd.DataFrame(
    ...     {
    ...         "age": np.random.normal(50, 10, 1000),
    ...         "children": np.random.poisson(2, 1000),
    ...         "sex": np.random.choice(
    ...             ["Male", "Female"],
    ...             size=1000,
    ...             p=[0.45, 0.55],
    ...         ),
    ...     }
    ... )
    >>>
    >>> syn_df = pd.DataFrame(
    ...     {
    ...         "age": np.random.normal(52, 12, 1000),
    ...         "children": np.random.poisson(3, 1000),
    ...         "sex": np.random.choice(
    ...             ["Male", "Female"],
    ...             size=1000,
    ...             p=[0.40, 0.60],
    ...         ),
    ...     }
    ... )
    >>>
    >>> orig_df.loc[:50, "age"] = np.nan # add missing
    >>> syn_df.loc[:30, "age"] = np.nan
    >>>
    >>> plots = plot_univariate_distributions(
    ...     orig_df=orig_df,
    ...     syn_df=syn_df,
    ...     save_path=None,
    ...     interactive=False,
    ... )
    >>> for fig in plots:
    ...     fig.show()
        
    """
    if not isinstance(orig_df, pd.DataFrame):
        raise ValueError(f"The original data should be a pandas DataFrame, got {type(orig_df)} instead.")
    if not isinstance(syn_df, pd.DataFrame):
        raise ValueError(f"The synthetic data should be a pandas DataFrame, got {type(syn_df)} instead.")
    if list(orig_df.columns) != list(syn_df.columns):
        raise ValueError("Original and synthetic datasets must have identical columns.")

    figures = [
        _plot_single_distribution(
            orig_df[column],
            syn_df[column],
            column,
        )
        for column in orig_df.columns
    ]

    if save_path is None and not interactive:
        return figures

    html_content = _build_html(figures)

    html_file = _write_html(html_content, save_path)

    if interactive:
        webbrowser.open(html_file.resolve().as_uri())
    
    return figures