import reproducibility as rep
import numpy as np

class DemoSynthMethod:
    def __init__(self,random_state = None):
        self.random_state = random_state
        pass

    def fit(self,X,y):

        self._seed = rep.Reseed.get_seed() if self.random_state is None else self.random_state
        self.random_state_ = rep.Reseed.get_rng(self._seed)

    def transform(self,X):
        rng = rep.Reseed.get_rng(self._seed)
        return rng.integers(0, 100, size=len(X))


class DemoSynthesiser:

    def __init__(self, seed, method = None):
        self.seed = seed
        self.method = method

    def fit(self,X):

        rep.Reseed.set_root_seed(self.seed)

        self.n_ = len(X)

        if self.method is None:
            self.method_ = DemoSynthMethod()

        self.method_.fit(X,y=None)


    def generate(self,n,seed=None):
        effective_seed = self.seed if seed is None else seed

        X0 = np.arange(0,n)
        with rep.Reseed(effective_seed):
            result = self.method_.transform(X0)

        return result

# From the users perspective:

syn = DemoSynthesiser(seed=6767)
X_obs = np.array([1,2,3,4,5,6,7,8,9])

syn.fit(X_obs)

s1 = syn.generate(n=10)
s2 = syn.generate(n=10)

print(f"seed = 6767, s1 = {s1}")
print(f"seed = 6767, s2 = {s2}")

s3 = syn.generate(n=10,seed=8989)
s4 = syn.generate(n=10,seed=8989)

print(f"seed = 8989, s3 = {s3}")
print(f"seed = 8989, s4 = {s4}")

s5 = syn.generate(n=10)
s6 = syn.generate(n=10)

print(f"seed = 6767, s5 = {s5}")
print(f"seed = 6767, s6 = {s6}")