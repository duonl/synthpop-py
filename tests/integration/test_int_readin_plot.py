def test_read_in_plotting_functions_correct_format():
    from synthpop.plotting import plot_univariate_distributions
    from synthpop.plotting import plot_spmse

def test_read_in_plotting_functions_all_():
    import synthpop.plotting as plotting

    assert plotting.__all__ == ["plot_spmse", "plot_univariate_distributions"]

    #Checks if it does not import internal functions
    assert not hasattr(plotting, "_make_matrix")
    assert not hasattr(plotting, "_get_colourscale")
    assert not hasattr(plotting, "_make_histograms")
    assert not hasattr(plotting, "_make_bars")
    assert not hasattr(plotting, "_plot_single_distribution")
    assert not hasattr(plotting, "_build_html")