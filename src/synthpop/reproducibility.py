import os.path
import secrets
import numpy as np
from numpy.random import SeedSequence

# Depending on the user for the root seed is suboptimal.
# Humans tend to not be so random. And forgetting to set a seed is too easy. 
# The seed 165204380881971836312753519450099508068 is a random number I would not have thought of.

# The proposed solution is to check if a seed has already been set (by existence of 'seed.txt').
# If the file exists, read its content and use it as a seed.
# If that file does not exists, generate a new seed of 128 bits and write it to 'seed.txt'
# This seeding happens when this file is imported, so the user does not need to remember to set a seed.



if os.path.isfile("seed.txt"):
    with open("seed.txt","r") as f:
        _root_seed = int(f.read())
else:
    _root_seed = secrets.randbits(128)
    with open("seed.txt","w") as f:
        f.write(str(_root_seed))

_root_rng = np.random.default_rng(_root_seed)
_seed_sequence = SeedSequence(_root_seed)

print(_root_rng.integers(0,100,10))


# now that we have a reproducible RNG, the rest of the package needs to use it.
# There are multiple possibilities here. 

# One possibility is to let the entire package use the same RNG instance. This would imply that multiple calls to fit or transform will give different results.
# Besides, it can introduce tricky to spot bugs when the RNG is altered (reset, for example). 
# It can also introduce problems with concurrency. 



# Another possibility is to use number drawn from the root RNG to seed RNGs in classes that need random numbers.
# This solves many of the problems with one "global" RNG. However, it places the burden of instanciating RNGs on classes that need random numbers.
# An possible advantage might be that it is probabily compatible with most other packages.
def get_seed():
    return _seed_sequence.spawn(1)[0]

# Another possibility is to provide RNGS to the other classes directly. This keeps the responsibility of creating RNGs in this module.
# It means we can alter the strategy for random numbers without altering the classes using them. 
# A disadvantage is that all classes are forced to accept the type of RNG that this module provides.
def get_rng():
    return _root_rng.spawn(1)[0]