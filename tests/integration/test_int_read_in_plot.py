def test_read_in_plotting_functions_correct_format():
    from synthpop.plotting import plot_univariate_distributions
    from synthpop.plotting import plot_spmse

    assert callable(plot_univariate_distributions)
    assert callable(plot_spmse)

def test_read_in_plotting_functions_all_():
    import synthpop.plotting as plotting

    assert plotting.__all__ == ["plot_spmse", "plot_univariate_distributions"]
    
    assert callable(plotting.plot_univariate_distributions)
    assert callable(plotting.plot_spmse)
    