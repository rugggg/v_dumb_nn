import numpy as np
from abc import ABC, abstractmethod

# implementing some base NN modules from scratch

class layer(ABC):

    def __init__(self):
        pass
     
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self):
        pass

class net():

    def __init__(self, name: str):
        self.name = name
        self.layers = []


class mult(layer):
    def __init__(self):
        self.dx = None
        self.dy = None
        self.x = None
        self.y = None

    def forward(self, x: np.array, y: np.array) -> np.array:
        self.x = x
        self.y = y
        z = x*y
        # if we have a gradient 
        return z
    
    def backward(self, dz: np.array) -> list[np.array, np.array]:
        dx = self.y * dz
        dy = self.x * dz
        return [dx, dy]


class fcl(layer):
    """fully connected layer - keeping activation func seperate for now"""
    def __init__(self, width: int, output_width: int, activation="relu"):
        self.width = width
        self.output_width = output_width
        self.W = 0.0001 * np.random.randn(self.width, self.output_width)
        self.b = np.zeros(self.output_width)
        self.activation=activation
        self.x = None
    
    def forward(self, x: np.array):
        self.x = x
        linear_out = np.dot(x, self.W) + self.b
        if self.activation == "relu":
            return np.maximum(linear_out,0)
        if self.activation == "softmax":
            return np.exp(linear_out)/np.sum(np.exp(linear_out))

    def backward(self, dz: np.array):
        m = self.x.shape[1]
        linear_back  = self.W.T
        dW = 1/m * np.dot(dz, self.x.T)
        db = 1/m * np.sum(dz, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dz)
        if self.activation == "relu":
            dz[self.x <= 0] = 0 # maybe need copy here?
            return dz
        if self.activation == "softmax":
            pass

'''
class sigmoid(layer):
    def forward(self, x: np.array):
        return 1 / (1+np.exp(-x))
    def backward(self, dz: np.array):
        pass

class relu(layer):
    def forward(self, x: np.array):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, dz: np.array):
        print(dz)
        dz[self.x <= 0] = 0 # maybe need copy here?
        return dz

class softmax_layer(layer):
    def forward(self, x):
        sm = np.exp(x)/np.sum(np.exp(x))
        return sm

    def backward(self):
        return super().backward()
    
def softmax(x):
    sm = np.exp(x)/np.sum(np.exp(x))
    return sm
'''
def ce_loss(y_pred, y_target):
    return -np.sum(y_target * np.log(y_pred)) # removed this idk if we need it / y_target.shape[0]

'''
def softmax_loss(y_pred, y_target):
    return ce_loss(softmax(y_pred), y_target)
'''
class Network():

    def __init__(self):
        self.layers = [
            fcl(10, 20, "relu"),
            fcl(20, 10, "softmax"),
        ]
        self.loss = ce_loss

    def forward(self, x, y):
        curr_input = x
        for idx, l in enumerate(self.layers):
            print(self.layers[idx], curr_input)
            curr_input = self.layers[idx].forward(curr_input)
        print(curr_input, y)
        l = self.loss(curr_input, y)
        return l # choosing not to return grads, they should be in the layers

    def backward(self, loss):
        print("----backprop----")
        grad = loss # correct? no lol

        for l in reversed(self.layers):
            grad = l.backward()
            # need to add updates to the layers
        print(grad)
        self.update() 
    
    def update(self):
        pass

if __name__ == "__main__":

    x = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    y = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

    nn = Network()
    loss = nn.forward(x, y)
    print(loss)
    nn.backward(loss)