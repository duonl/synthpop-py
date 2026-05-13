import numpy as np

str_dtype = np.dtypes.StringDType(na_object=np.nan)

def to_stringdtype_array(x):
    if isinstance(x,list):
        return np.array(x,dtype=str_dtype)
    return x.astype(str_dtype,copy=False)
