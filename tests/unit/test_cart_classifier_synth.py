


from synthpop.methods.cart_synth import TreeClassifierMethod


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5

def test_datatype_sanity():
    obj = TreeClassifierMethod()
    assert type(obj) is TreeClassifierMethod