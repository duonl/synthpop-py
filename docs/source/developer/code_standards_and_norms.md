# Code standards, norms, and conventions

This document describes when code is good enough to be accepted in this code base. 
It also provides tools to check some of these norms. 
See this document as a tie-breaker when there are multiple equally well options, but a consistent choice is preferred. 

## Language and form
- British english
- [PEP8](https://peps.python.org/pep-0008/)
- Use the term "original data" to refer to the data that is not synthetic. 
- Use typehints where appropriate. 

## Design and develop principles
- Test Driven Development, with a code coverage of at least 80%. 
- [SOLID](https://en.wikipedia.org/wiki/SOLID)
- It should always be possible to generate synthetic data within 5 lines of code. 
- All classes that inherit from BaseEstimator should follow [the developer guides of scikit-learn](https://scikit-learn.org/stable/developers/develop.html)
- Use numpy arrays where possible. 
- The user of this package should be able to use pandas

A synthesis method should accept pandas dataframes/series and heterogeneous data.
Making the data homogeneous should happen within the synthesis method.
It is best to use numpy within a synthesis method. See [#45](https://github.com/Synthpop-data/synthpop-py/issues/45).




## Standard topics in testing

The things that should be tested depend mostly on what is being made, and follow from the requirements.
However, there are a few things that need to be tested in many situations. Take the following list as inspiration, not a set in stone rule.

- Is the return value of the correct type and shape:
    - Is the returned value a list, numpy array, pandas dataframe, pandas series?
    - Are the columns/key-value pairs/items of the correct type?
    - Are any names (of columns or keys) correct?
- For estimators (in the sense of scikit-learn), there is a [pre-made set of test](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks) for compatibility met scikit-learn. Also see [this](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator). You can look [here](https://scikit-learn.org/stable/api/sklearn.utils.html) for more general tools.
- Obvious cases, for example:
    - taking the mean of a set with one value. 
    - sorting an already sorted list.
    - metric of 2 identical datasets. 
- Use cases. Tests that reflect how the code is supposed to be used. 
- Edge cases:
    - Dataframes that have no columns or have no rows.
    - Numeric values that are zero or below zero. 
    - For every occurrence of "==","<", "<=",">",">=" there are edge cases. 
    - For every numeric division there is a possibility to have a zero division. 
    - Wherever there is a "==" between numerical values, there could be floating point errors. (try 0.1+0.2 == 0.3)
- Are the inputs assumed to be related to each other? Test what happens when they are not. For example:
    - One input specifies the output file, a second input specifies output format. What happens if an output format is given but no  output file?
    - One input specifies the number of principle components, a second input specifies a threshold to select principle components. What happens when both are given?

## Useful tools for testing and code quality



- Code Spell Checker extension for vscode.
- autopep8 extension for vscode. 
- pytest-cov for code coverage of the unit tests. (installed if you installed this package with the dev dependency group)
- pylint for code feedback and code analysis. 
    