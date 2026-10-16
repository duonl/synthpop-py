Utilities
=========

.. _string-dtype:

String arrays with missing values
---------------------------------

.. py:data:: synthpop.utils.str_dtype

   A shared instance of ``numpy.dtypes.StringDType(na_object=np.nan)``:
   NumPy's variable-width string dtype, configured to retain ``np.nan`` as a
   missing value rather than turn it into the text ``"nan"``.

Use this dtype when constructing categorical NumPy arrays for the low-level
encoders and missing-value handlers shown in the examples:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from synthpop.utils import str_dtype

   values = np.array(["cat", np.nan, "dog"], dtype=str_dtype)
   assert pd.isna(values).tolist() == [False, True, False]
   assert values[0] == "cat"

By contrast, ``np.array(["cat", np.nan], dtype=str)`` uses NumPy's fixed-width
Unicode dtype and converts the missing value to the ordinary string ``"nan"``.
That loses the distinction between missing data and a category whose label is
literally ``"nan"``. ``str_dtype`` also avoids fixed-width string truncation
when longer category labels are assigned.

This is a dtype object, not a function or a synthesis method. Pass it as
``dtype=str_dtype`` to ``np.array`` or use ``array.astype(str_dtype)``. The
configuration is specifically for ``np.nan``; a different ``StringDType``
configuration, such as ``na_object=None``, is not interchangeable in the
low-level validation helpers. Keep genuinely numeric variables numeric:
converting numbers to strings makes them categorical.
