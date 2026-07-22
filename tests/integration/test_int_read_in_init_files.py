# --------- __init__ files for users ---------

def test_read_in_synthpop_all_():
    import synthpop

    assert synthpop.__all__ == [
        "data_processing",
        "methods",
        "plotting",
        "utility_metrics",
        "Synthesiser"
    ]

    assert callable(synthpop.Synthesiser)
    assert synthpop.__version__ == "1.0.0"

def test_read_in_plotting_functions_all_():
    import synthpop.plotting as plotting

    assert plotting.__all__ == [
        "plot_spmse",
        "plot_univariate_distributions"
    ]

    assert callable(plotting.plot_univariate_distributions)
    assert callable(plotting.plot_spmse)


def test_read_in_utility_metric_functions_all_():
    import synthpop.utility_metrics as util_metrics

    assert util_metrics.__all__ == [
        "pairwise_spmse",
    ]

    assert callable(util_metrics.pairwise_spmse)


def test_read_in_methods_functions_all_():
    import synthpop.methods as methods

    assert methods.__all__ == [
        "CopyMethod",
        "SampleMethod",
        "BaseSynthMethod",
        "CartMethod",
        "tune_cart",
        "TreeRegressorMethod",
        "TreeClassifierMethod"
    ]

    assert callable(methods.CopyMethod)
    assert callable(methods.SampleMethod)
    assert callable(methods.CartMethod)
    assert callable(methods.tune_cart)
    assert callable(methods.TreeRegressorMethod)
    assert callable(methods.TreeClassifierMethod)


def test_read_in_data_processing_functions_all_():
    import synthpop.data_processing as dp

    assert dp.__all__ == [
        "PCAEncoder",
        "MeanEncoder",
        "MissingValuePredictor",
        "ReplaceMissingWithValue",
    ]

    assert callable(dp.PCAEncoder)
    assert callable(dp.MeanEncoder)
    assert callable(dp.MissingValuePredictor)
    assert callable(dp.ReplaceMissingWithValue)