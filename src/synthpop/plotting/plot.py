"""
This module contains functions to visually inspect synthetic data and evaluate its quality. 
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html


def plot_univariate_distributions(
        obs_df: pd.DataFrame, 
        syn_df: pd.DataFrame, 
        saving_location: str | None =  None,
        interactive: bool = True,
        ) -> None:
    """
    Plot comparisons of the univariate distribution between the observed and synthetic data.
    
    :param obs_df: Original/observed dataset.
    :param syn_df: Synthetic dataset.
    :param saving_location: Folder where the HTML file should be written. If None,
        the plots will not be saved.
    :param open_browser: In headless running set to False

    :return: None
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise ValueError(f"The observed data should be a pandas DataFrame, got {type(obs_df)} instead.")
    if not isinstance(syn_df, pd.DataFrame):
        raise ValueError(f"The synthetic data should be a pandas DataFrame, got {type(syn_df)} instead.")
    if list(obs_df.columns) != list(syn_df.columns):
        raise ValueError("Observed and synthetic datasets must have identical columns.")

    html_parts: list[str] = [
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

    for column in obs_df.columns:

        obs = obs_df[column]
        syn = syn_df[column]

        obs_missing = int(obs.isna().sum())
        syn_missing = int(syn.isna().sum())

        fig = go.Figure()

        # ------------------------------------------------------------------
        # Numeric variables
        # ------------------------------------------------------------------
        if pd.api.types.is_numeric_dtype(obs):

            obs_non_missing = obs.dropna()
            syn_non_missing = syn.dropna()

            combined = pd.concat(
                [obs_non_missing, syn_non_missing],
                ignore_index=True,
            )

            if len(combined) > 0:

                is_integer = (
                    pd.api.types.is_integer_dtype(obs)
                    and pd.api.types.is_integer_dtype(syn)
                )

                if is_integer:
                    min_val = int(combined.min())
                    max_val = int(combined.max())

                    fig.add_trace(
                        go.Histogram(
                            x=obs_non_missing,
                            name="Observed",
                            histnorm="probability density",
                            xbins=dict(
                                start=min_val - 0.5,
                                end=max_val + 0.5,
                                size=1,
                            ),
                            opacity=0.6,
                        )
                    )

                    fig.add_trace(
                        go.Histogram(
                            x=syn_non_missing,
                            name="Synthetic",
                            histnorm="probability density",
                            xbins=dict(
                                start=min_val - 0.5,
                                end=max_val + 0.5,
                                size=1,
                            ),
                            opacity=0.6,
                        )
                    )

                else:
                    bins = np.histogram_bin_edges(
                        combined.to_numpy(),
                        bins="auto",
                    )

                    fig.add_trace(
                        go.Histogram(
                            x=obs_non_missing,
                            name="Observed",
                            histnorm="probability density",
                            xbins=dict(
                                start=float(bins[0]),
                                end=float(bins[-1]),
                                size=float(bins[1] - bins[0]),
                            ),
                            opacity=0.6,
                        )
                    )

                    fig.add_trace(
                        go.Histogram(
                            x=syn_non_missing,
                            name="Synthetic",
                            histnorm="probability density",
                            xbins=dict(
                                start=float(bins[0]),
                                end=float(bins[-1]),
                                size=float(bins[1] - bins[0]),
                            ),
                            opacity=0.6,
                        )
                    )

            fig.update_layout(
                title=f"Distribution comparison: {column}",
                xaxis_title=column,
                yaxis_title="Density",
                barmode="overlay",
            )

        # ------------------------------------------------------------------
        # Categorical / non-numeric variables
        # ------------------------------------------------------------------
        else:

            obs_counts = (
                obs.fillna("<MISSING>")
                .value_counts(normalize=True, dropna=False)
            )

            syn_counts = (
                syn.fillna("<MISSING>")
                .value_counts(normalize=True, dropna=False)
            )

            levels = sorted(
                set(obs_counts.index.astype(str))
                | set(syn_counts.index.astype(str))
            )

            obs_density = [
                float(obs_counts.get(level, 0.0))
                for level in levels
            ]

            syn_density = [
                float(syn_counts.get(level, 0.0))
                for level in levels
            ]

            fig.add_trace(
                go.Bar(
                    x=levels,
                    y=obs_density,
                    name="Observed",
                    customdata=[
                        int(
                            (
                                obs.fillna("<MISSING>")
                                == level
                            ).sum()
                        )
                        for level in levels
                    ],
                    hovertemplate=(
                        "Level=%{x}<br>"
                        "Density=%{y:.4f}<br>"
                        "Count=%{customdata}"
                        "<extra>Observed</extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Bar(
                    x=levels,
                    y=syn_density,
                    name="Synthetic",
                    customdata=[
                        int(
                            (
                                syn.fillna("<MISSING>")
                                == level
                            ).sum()
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
            )

            fig.update_layout(
                title=f"Distribution comparison: {column}",
                xaxis_title=column,
                yaxis_title="Density",
                barmode="group",
            )

        # ------------------------------------------------------------------
        # Missing value annotation
        # ------------------------------------------------------------------
        fig.add_annotation(
            text=(
                f"Missing values - "
                f"Observed: {obs_missing}, "
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

        html_parts.append(
            to_html(
                fig,
                include_plotlyjs="cdn",
                full_html=False,
            )
        )

    html_parts.append("</body></html>")

    html_content = "\n".join(html_parts)

    if saving_location is None:
        if interactive:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".html",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_file.write(html_content)

            webbrowser.open(Path(temp_file.name).resolve().as_uri())
        return
    
    output_dir = Path(saving_location)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "univariate_distribution_comparison.html"
    output_file.write_text(html_content, encoding="utf-8")

    if interactive:
        webbrowser.open(output_file.resolve().as_uri())
    

def plot_spmse(spmse: pd.DataFrame, target_file: str | None) -> None:
    """
    Plot the standardised propensity mean squared error.
    
    :param spmse: The standardised propensity mean squared error values
    :param target_file: File name to save the image of the plot

    :return: None
    """
    return None