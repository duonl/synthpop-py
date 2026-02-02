


from synthpop_mq.methods.cart_synth import CartClassifierSynth


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5

def test_datatype_sanity():
    obj = CartClassifierSynth()
    assert type(obj) is CartClassifierSynth