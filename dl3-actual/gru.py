import numpy as np
import itertools


class Sigmoid:
    # Magic. Do not touch.
    def __init__(self):
        pass

    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res

    def backward(self):
        return self.res * (1-self.res)

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    # Magic. Do not touch.
    def __init__(self):
        pass

    def forward(self, x):
        self.res = np.tanh(x)
        return self.res

    def backward(self):
        return 1 - (self.res**2)

    def __call__(self, x):
        return self.forward(x)


class GRU_Cell:
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d

        #Method to the madness
        np.random.seed(11785)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wh = np.random.randn(h, h)

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wx = np.random.randn(h, d)

        # ...

        self.r_act = Sigmoid()
        self.h_act = Tanh()

    def forward(self, x, h):
        #You may want to store some of the calculated values
        raise NotImplementedError()

    def backward(self, delta):
        #Use the saved values from forward

        #Hint:
        #dWh = ??
        #dWx = ??
        #dWrh = ??
        #dWrx = ??
        #dWzh = ??
        #dWzx = ??
        #dx = ??
        #dh = ??
        #And also the derivates of any intermediate values
        raise NotImplementedError()
