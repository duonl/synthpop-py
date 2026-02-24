"""
This module contains utilities to visually inspect synthetic data and evaluate its quality. 
"""
import pandas as pd

def plot_univariate_distributions(obs_df: pd.DataFrame, syn_df: pd.DataFrame, target_folder: str | None) -> None:
    """
    Plot comparisons of the univariate distribution between the observed and synthetic data
    
    :param obs_df: The observed data
    :param syn_df: The synthetic data
    :param target_folder: Folder where images need to be saved

    :return: None
    """
    return None

def plot_spmse(spmse: pd.DataFrame, target_file: str | None) -> None:
    """
    Plot the standardised propensity mean squared error.
    
    :param spmse: The standardised propensity mean squared error values
    :param target_file: File name to save the image of the plot

    :return: None
    """
    return None