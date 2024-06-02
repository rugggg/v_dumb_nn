import numpy as np
import pandas as pd
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
    def __init__(self, width: int, output_width: int, activation="relu", name=None):
        self.width = width
        self.output_width = output_width
        # self.W = 0.0001 * np.random.randn(self.output_width, self.width)
        self.W = np.random.standard_normal((self.output_width, self.width)) * np.sqrt(2/self.width)
        self.b = np.zeros(self.output_width)
        self.activation=activation
        self.name = name
        self.x = None # hold the input, probably dont need
        self.activation_cache = None # hold the activation output
        self.linear_cache = None # hold the linear layer output before activation
        self.dW = None # derivative wrt weights
        self.db = None # derivative wrt bias
        self.dz = None # derivative from upstream layers and originally the loss

    def forward(self, x: np.array):
        self.x = x # hold the inputs to this layer for backprop
        #self.linear_cache = np.dot(self.W, x) + self.b # z is the linear activation output, z == linear cache
        self.linear_cache = self.W @ x + self.b
        # then do the non-linearity activation function of choice on that linear activation output
        if self.activation == "relu":
            self.activation_cache = np.maximum(self.linear_cache ,0)
        if self.activation == "softmax":
            self.activation_cache =  np.exp(self.linear_cache)/np.sum(np.exp(self.linear_cache))
        return self.activation_cache

    def backward(self, grad: np.array):
        # in a backward pass, we need to get the gradients with respect to x, W, and b

        # get the gradient of the 
        if self.activation == "relu":
            grad[self.activation_cache <= 0] = 0 # maybe need copy here?
            self.dz = grad
        if self.activation == "softmax":
            self.dz = self.activation_cache - grad # grad == Y when you have softmax, it's last layer
        # linear backward
        m = self.activation_cache.shape[0]
        m = 1 # for single batch size hack check
        print(m, self.dz.shape, self.activation_cache.T.shape)
        self.dW = 1/m * np.dot(self.dz, self.linear_cache.T)
        self.db = 1/m * np.sum(self.dz, keepdims=True)
        self.dA_prev = np.dot(self.W.T, self.dz)
        self.update()
        return self.dA_prev

    def update(self, lr=1.1):
        # so class has it's own grads already
        # print("updating with grad", self.dW, self.W[0])
        self.W = self.W - self.dW*lr
        self.b = self.b - self.db*lr
        # print("set with grad", self.W[0])

    

def ce_loss(y_pred, y_target):
    return -np.sum(y_target * np.log(y_pred)) # removed this idk if we need it / y_target.shape[0]

class Network():

    def __init__(self):
        self.layers = [
            fcl(10, 20, "relu", "relu layer"),
            fcl(20, 10, "softmax", "softmax layer"),
        ]
        self.loss = ce_loss

    def forward(self, x, y):
        curr_input = x
        for idx, l in enumerate(self.layers):
            curr_input = self.layers[idx].forward(curr_input)
        l = self.loss(curr_input, y)
        return curr_input, l # choosing not to return grads, they should be in the layers

    def backward(self, y_pred, y_target):
        print("----backprop----")
        grad = y_pred
        for l in reversed(self.layers):
            if l.activation == "softmax":
                grad = l.backward(y_target)
            else:
                grad = l.backward(grad)


def load_data(path_to_file):
    loaded = pd.read_csv(path_to_file)
    labels = loaded['label']
    data = loaded.loc[:, loaded.columns != 'label']
    nb_classes = 10
    targets = labels.to_numpy().reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return data.to_numpy(), one_hot_targets

    


if __name__ == "__main__":
    
    # data for minibatch
    x = np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
    y = np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
    # tiny test check
    #x = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    #y = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

    #x, y =load_data("./data/train.csv")
    #print(x.shape, y.shape)
    
    nn = Network()

    for it in range(10):
        #for sample_idx in range(300):
        #y_pred, loss = nn.forward(x[sample_idx], y[sample_idx])
        y_pred, loss = nn.forward(x, y)
        print("loss:", loss, "pred:", y_pred)
        nn.backward(y_pred, y)