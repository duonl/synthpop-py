Utilities
=========

This module contains utility functions used throughout synthpop-py.

.. py:data:: synthpop.utils.str_dtype

   A shared instance of ``numpy.dtypes.StringDType(na_object=np.nan)``.

.. _string-dtype:

Examples
--------

With ``dtype=str``, a missing value becomes the string ``"nan"``. It can no
longer be distinguished from a category whose label is literally ``"nan"``:

>>> import numpy as np
>>> import pandas as pd
>>> from synthpop.utils import str_dtype
>>> fixed = np.array(["cat", np.nan, "nan"], dtype=str)
>>> fixed.tolist()
['cat', 'nan', 'nan']
>>> pd.isna(fixed).tolist()
[False, False, False]

Constructing the array with ``str_dtype`` preserves that distinction, so
missing-value handlers can still identify the missing entry:

>>> values = np.array(["cat", np.nan, "nan"], dtype=str_dtype)
>>> values.tolist()
['cat', nan, 'nan']
>>> pd.isna(values).tolist()
[False, True, False]

``str_dtype`` is a dtype object, not a function. An existing object array can
also be converted with ``astype`` while it still contains the original missing
value:

>>> original = np.array(["cat", np.nan, "nan"], dtype=object)
>>> converted = original.astype(str_dtype)
>>> converted.tolist()
['cat', nan, 'nan']
>>> pd.isna(converted).tolist()
[False, True, False]

Converting ``fixed`` instead would not recover the missing value: its entries
are already strings.
