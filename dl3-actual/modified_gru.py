import numpy as np
import sys

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
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wrh = np.random.randn(self.hidden_size, self.hidden_size)
        self.Wzh = np.random.randn(self.hidden_size, self.hidden_size)
        self.Wh = np.random.randn(self.hidden_size, self.hidden_size)
        self.Wrx = np.random.randn(self.hidden_size, self.input_size)
        self.Wzx = np.random.randn(self.hidden_size, self.input_size)
        self.Wx = np.random.randn(self.hidden_size, self.input_size)

    def forward(self, input, h_prev):
        self.input = input
        self.h_prev = h_prev
        sigmoid = Sigmoid()
        tanh = Tanh()

        self.zt = sigmoid(np.dot(self.Wzh, self.h_prev) + np.dot(self.Wzx, self.input))
        self.rt = sigmoid(np.dot(self.Wrh, self.h_prev) + np.dot(self.Wrx, self.input))
        self.h_hat = tanh(np.dot(self.Wh, np.multiply(self.rt, self.h_prev)) + np.dot(self.Wx, self.input))
        self.ht = np.multiply((1 - self.zt), self.h_hat) + np.multiply(self.zt, self.h_prev)
        return self.ht

    def backward(self, delta):
        # d9 error
        # d1 = np.multiply(self.zt, delta)
        # d2 = np.multiply(self.h_prev, delta)
        # d3 = np.multiply(self.h_hat, delta)
        # d4 = np.multiply(-1, d3)
        # d5 = d2 + d4
        # d6 = np.multiply((1 - self.zt), delta)
        # d7 = np.multiply(d5, np.multiply(self.zt, (1 - self.zt)))
        # d8 = np.multiply(d6, (1 - np.square(self.h_hat)))
        # d9 = np.dot(d8, self.Wx.T)
        # d10 = np.dot(d8, self.Wh.T)
        # d11 = np.dot(d7, self.Wzx.T)
        # d12 = np.dot(d7, self.Wzh.T)
        # d14 = np.multiply(d10, self.rt)
        # d15 = np.multiply(d10, self.h_prev)
        # d16 = np.multiply(d15, np.multiply(self.rt, (1 - self.rt)))
        # d13 = np.dot(d16, self.Wrx.T)
        # d17 = np.dot(d16, self.Wrh.T)
        # dx = d9 + d11 + d13
        # dh_prev = d12 + d14 + d1 + d17
        # dWrx = np.dot(self.input.T, d16)
        # dWzx = np.dot(self.input.T, d7)
        # dWx = np.dot(self.input.T, d8)
        # dWrh = np.dot(self.h_prev.T, d16)
        # dWzh = np.dot(self.h_prev.T, d7)
        # dWh = np.dot(np.multiply(self.h_prev, self.rt).T, d8)

        rt = self.rt
        rt = rt.reshape((-1, 1))
        Wr = self.Wzh
        h_prev = self.h_prev
        h_prev = h_prev.reshape((-1, 1))
        Ur = self.Wzx
        xt = self.input
        xt = xt.reshape((-1, 1))
        zt = self.zt
        zt = zt.reshape((-1, 1))
        Wz = self.Wrh
        Uz = self.Wrx
        h_hat = self.h_hat
        h_hat = h_hat.reshape((-1, 1))
        Wh = self.Wh
        Uh = self.Wx
        ht = self.ht
        ht = ht.reshape((-1, 1))

        d0 = delta
        d1 = np.multiply(zt, d0)
        d2 = np.multiply(h_prev, d0)
        d3 = np.multiply(h_hat, d0)
        d4 = np.multiply(-1, d3)
        d5 = d2 + d4
        d6 = np.multiply((1 - zt), d0)
        d7 = np.multiply(d5, np.multiply(zt, (1 - zt)))
        d8 = np.multiply(d6, (1 - np.square(h_hat)))
        d9 = np.dot(d8, Uh)#Uh.T
        d10 = np.dot(d8, Wh.T)
        d11 = np.dot(d7, Uz)#Uz.T
        d12 = np.dot(d7, Wz.T)
        d14 = np.multiply(d10, rt)
        d15 = np.multiply(d10, h_prev)
        d16 = np.multiply(d15, np.multiply(rt, (1 - rt)))
        d13 = np.dot(d16, Ur)#Ur.T
        d17 = np.dot(d16, Wr.T)

        dx = d9 + d11 + d13
        dh_prev = d12 + d14 + d1 + d17
        dUr = np.dot(delta, d16)
        dUz = np.dot(delta, d7)
        dUh = np.dot(delta, d8)
        dWr = np.dot(h_prev.T, d16)
        dWz = np.dot(h_prev.T, d7)
        dWh = np.dot(np.multiply(h_prev, rt).T, d8)

        self.dWh = dWh
        self.dWx = dUh
        self.dWrh = dWr
        self.dWrx = dUz
        self.dWzh = dWr
        self.dWzx = dUz
        self.dx = dx
        self.dh = dh_prev

        #Hint:
        #dWh = ?? - x
        #dWx = ?? - x
        #dWrh = ?? - x
        #dWrx = ?? - x
        #dWzh = ?? - x
        #dWzx = ?? - x
        #dx = ?? - x
        #dh = ?? - x
        #And also the derivates of any intermediate values
        #raise NotImplementedError()

        return dx
