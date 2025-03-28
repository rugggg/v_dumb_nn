import jax
import jax.numpy as jnp
import math
import plotext as plt
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Callable


class JaxLayer(ABC):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self):
        pass


class JaxFCL(JaxLayer):
    """Fully connected layer implemented with JAX"""
    def __init__(self, width: int, output_width: int, activation="relu", name=None):
        self.width = width
        self.output_width = output_width
        
        # Initialize with He initialization
        key = jax.random.PRNGKey(0)
        self.W = jax.random.normal(key, (self.output_width, self.width)) * jnp.sqrt(2/self.width)
        self.b = jnp.zeros((self.output_width, 1))
        self.activation = activation
        self.name = name

        # Caches for backward pass
        self.x = None
        self.activation_cache = None
        self.linear_cache = None
        self.dW = None
        self.db = None
        self.dz = None

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.x = x
        self.linear_cache = jnp.dot(self.W, x) + self.b
        
        if self.activation == "relu":
            self.activation_cache = jnp.maximum(self.linear_cache, 0)
        elif self.activation == "softmax":
            self.activation_cache = jax_softmax(self.linear_cache)
        return self.activation_cache

    def backward(self, grad: jnp.ndarray) -> jnp.ndarray:
        if self.activation == "relu":
            # Create boolean mask where activation <= 0
            mask = self.linear_cache <= 0
            # Using JAX's where function to apply the mask
            self.dz = jnp.where(mask, 0, grad)
        elif self.activation == "softmax":
            self.dz = grad
            
        m = self.x.shape[1]
        self.dW = (1/m) * jnp.dot(self.dz, self.x.T)
        self.db = (1/m) * jnp.sum(self.dz, axis=1, keepdims=True)
        dA_prev = jnp.dot(self.W.T, self.dz)
        
        # Update weights and biases
        self.update()
        return dA_prev

    def update(self, lr=0.1):
        assert self.W.shape == self.dW.shape
        assert self.b.shape == self.db.shape
        
        # In JAX, arrays are immutable, so we need to create new arrays
        self.W = self.W - lr * self.dW 
        self.b = self.b - lr * self.db


def jax_softmax(x: jnp.ndarray) -> jnp.ndarray:
    """Compute the softmax using JAX's stable implementation"""
    # Subtracting the max for numerical stability
    x_max = jnp.max(x, axis=0, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=0, keepdims=True)


def jax_cross_entropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cross-entropy loss using JAX
    Inputs are shape (feat, batch)
    """
    # Adding small epsilon for numerical stability
    eps = 1e-10
    y_pred_safe = jnp.clip(y_pred, eps, 1.0 - eps)
    return -jnp.sum(y_true * jnp.log(y_pred_safe), axis=0)


class JaxNetwork:
    def __init__(self, input_dim=784, hidden_dims=[128, 32], output_dim=10):
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for i, dim in enumerate(hidden_dims):
            layers.append(JaxFCL(prev_dim, dim, "relu", f"relu_layer_{i}"))
            prev_dim = dim
            
        # Create output layer
        layers.append(JaxFCL(prev_dim, output_dim, "softmax", "softmax_layer"))
        
        self.layers = layers

    def forward(self, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through the network with loss computation"""
        curr_input = x
        for layer in self.layers:
            curr_input = layer.forward(curr_input)
            
        # Compute loss and gradient
        loss = jax_cross_entropy(y, curr_input)
        loss_grad = curr_input - y  # Gradient of softmax + cross-entropy
        
        return curr_input, loss, loss_grad

    def backward(self, loss_grad: jnp.ndarray):
        """Backward pass through the network"""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


def load_data_jax(train_file: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load MNIST data and convert to JAX arrays"""
    import numpy as np  # Use numpy for initial loading then convert to JAX
    
    raw = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
    
    Y = raw[:, 0]  # First column contains labels
    X = raw[:, 1:785]  # Rest is pixel data
    Y = Y.reshape(raw.shape[0], 1)
    
    # Normalize and convert to JAX arrays
    train_x = jnp.array(X.T/255.)
    train_y = jnp.array(Y.T)
    y_train_hot = jnp.squeeze(jnp.eye(10)[train_y.astype(int)])
    
    return train_x, y_train_hot.T


if __name__ == "__main__":
    X, Y = load_data_jax("data/train.csv")
    nn = JaxNetwork()
    losses = []
    
    for it in range(200):
        y_pred, loss, loss_grad = nn.forward(X, Y)
        mean_loss = jnp.mean(loss).item()
        
        print(f"loss: {mean_loss}", end='\r')
        losses.append(mean_loss)
        
        if math.isnan(mean_loss):
            print("NAN LOSS! Failed")
            break
            
        nn.backward(loss_grad)
        
    plt.theme("sahara")
    plt.plot_size(100, 30)
    plt.plot(losses)
    plt.title("Training Loss per Epoch (JAX Implementation)")
    plt.show()