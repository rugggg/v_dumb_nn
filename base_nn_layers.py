import numpy as np
import math
import plotext as plt
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


class fcl(layer):
    """fully connected layer - keeping activation func seperate for now"""
    def __init__(self, width: int, output_width: int, activation="relu", name=None):
        self.width = width
        self.output_width = output_width
        # self.W = np.random.randn(self.output_width, self.width)
        self.W = np.random.standard_normal((self.output_width, self.width)) * np.sqrt(2/self.width)
        self.b = np.zeros((self.output_width,1))
        self.activation=activation
        self.name = name

        self.x = None # hold the input to this layer
        self.activation_cache = None # hold the activation output
        self.linear_cache = None # hold the linear layer output before activation
        self.dW = None # derivative wrt weights
        self.db = None # derivative wrt bias
        self.dz = None # derivative from upstream layers

    def forward(self, x: np.array):
        self.x = x # hold the inputs to this layer for backprop
        self.linear_cache = np.dot(self.W, x) + self.b # z is the linear activation output, z == linear cache
        # self.linear_cache = self.W @ x + self.b
        # then do the non-linearity activation function of choice on that linear activation output
        if self.activation == "relu":
            self.activation_cache = np.maximum(self.linear_cache, 0)
        if self.activation == "softmax":
            self.activation_cache = softmax(self.linear_cache)
        return self.activation_cache

    def backward(self, grad: np.array):
        # in a backward pass, we need to get the gradients with respect to x, W, and b

        # get the gradient of the 
        if self.activation == "relu":
            grad[self.activation_cache <= 0] = 0 # maybe need copy here?
            self.dz = grad
        if self.activation == "softmax":
            self.dz = grad
        # linear backward
        m = self.x.shape[1]
        self.dW = 1/m * np.dot(self.dz, self.x.T) # is this right? 
        self.db = 1/m * np.sum(self.dz, axis=1, keepdims=True)
        self.dA_prev = np.dot(self.W.T, self.dz)
        self.update()
        return self.dA_prev

    def update(self, lr=0.1):
        assert self.W.shape == self.dW.shape
        assert self.b.shape == self.db.shape
        self.W = self.W - self.dW*lr 
        self.b = self.b - self.db*lr 
    

def softmax(x):
    """Compute the softmax of vector x. on input that is shape (feat, batch), hence axis=0"""
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


def cross_entropy(y_true, y_pred):
    """
    Compute the cross-entropy of the predicted probability distribution y_pred and the true distribution y_true.
    on input that is shape (feat, batch), hence axis=0
    """
    assert y_true.shape == y_pred.shape
    return -np.sum(y_true * np.log(y_pred), axis=0)


class Network():

    def __init__(self):
        self.layers = [
            fcl(784, 128, "relu", "relu layer"),
            fcl(128, 32, "relu", "relu64layer"),
            fcl(32, 10, "softmax", "softmax layer"),
        ]

    def forward(self, x, y):
        curr_input = x
        for idx, l in enumerate(self.layers):
            curr_input = self.layers[idx].forward(curr_input)
        l = cross_entropy(y, curr_input)
        l_grad = curr_input - y
        return curr_input, l, l_grad

    def backward(self, loss_grad):
        grad = loss_grad
        for l in reversed(self.layers):
            if l.activation == "softmax":
                grad = l.backward(grad)
            else:
                grad = l.backward(grad)


def load_data(train_file, test_file=None):
    raw = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

    Y = raw[:,0]     # first column of raw data are image labels
    X = raw[:,1:785]   # rest of the pixel data is in columns 1 to 785
    Y = Y.reshape(raw.shape[0],1)

    train_x = (X.T)/255.
    train_y = Y.T
    y_train_hot = np.squeeze(np.eye(10)[train_y])
    return train_x, y_train_hot.T


def parse_args():
    parser = argparse.ArgumentParser(
                    prog='A rather poor implementation of a basic Fully Connected NN for MNIST',
                    description='You can train, or you can run a prediction on a saved model',
                    epilog='Text at the bottom of help'
            )
    parser.add_argument('-m', '--mode', options=['train', 'predict'])  # on/off flag



def plot_losses(losses, step, total_steps):
    plt.clear_terminal()
    plt.clear_data()
    plt.theme("sahara")
    plt.plot_size(100, 30)
    
    # Create x-axis values for steps
    x_values = list(range(1, len(losses) + 1))
    
    # Set axis limits to show a good view of the data
    y_min = min(losses) * 0.9 if losses else 0
    y_max = max(losses) * 1.1 if losses else 3
    plt.xlim(1, total_steps)
    plt.ylim(y_min, y_max)
    
    # Plot with labels and grid
    plt.plot(x_values, losses)
    plt.grid(True)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Training Loss per Epoch (Step {step}/{total_steps})")
    plt.show()

if __name__ == "__main__":
    X, Y = load_data("data/train.csv")
    nn = Network()
    losses = []
    total_steps = 200

    for it in range(total_steps):
        y_pred, loss, loss_grad = nn.forward(X, Y)
        current_loss = loss.mean()
        print(f"loss: {current_loss:.6f} (Step {it+1}/{total_steps})", end='\r')
        losses.append(current_loss)
        
        if math.isnan(current_loss):
            print("\nNAN LOSS! Failed")
            break
            
        nn.backward(loss_grad)
        
        # Show plot every 5 steps
        if (it + 1) % 5 == 0 or it == 0 or it == total_steps - 1:
            plot_losses(losses, it + 1, total_steps)
    
    # Final plot with enhanced visualization
    plt.clear_terminal()
    plt.clear_data()
    plt.theme("sahara")
    plt.plot_size(100, 30)
    
    # Create x-axis values for steps
    x_values = list(range(1, len(losses) + 1))
    
    # Set axis limits for final plot
    y_min = min(losses) * 0.9
    y_max = max(losses) * 1.1
    plt.xlim(1, len(losses))
    plt.ylim(y_min, y_max)
    
    # Plot with labels and grid
    plt.plot(x_values, losses)
    plt.grid(True)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Final Training Loss per Epoch (NumPy Implementation)")
    plt.show()
