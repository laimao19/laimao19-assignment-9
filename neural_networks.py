import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = self.activation(activation)
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def activation(self, name):
        if name == "tanh":
            return np.tanh
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Invalid activation function")

    def activation_derivative(self, x):
        if self.activation_fn == np.tanh:
            return 1 - np.tanh(x)**2
        elif self.activation_fn.__name__ == '<lambda>' and 'maximum' in str(self.activation_fn):  # ReLU
            return (x > 0).astype(float)
        else:  # sigmoid
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation_fn(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
 import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = self.activation(activation)
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def activation(self, name):
        if name == "tanh":
            return np.tanh
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Invalid activation function")

    def activation_derivative(self, x):
        if self.activation_fn == np.tanh:
            return 1 - np.tanh(x)**2
        elif self.activation_fn.__name__ == '<lambda>' and 'maximum' in str(self.activation_fn):  # ReLU
            return (x > 0).astype(float)
        else:  # sigmoid
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation_fn(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    for _ in range(10):  # Perform multiple training steps
        mlp.forward(X)
        mlp.backward(X, y)

    # === Hidden Space Visualization ===
    hidden_features = mlp.activation_fn(np.dot(X, mlp.W1) + mlp.b1)  # Apply activation
    ax_hidden.clear()
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap="bwr",
        alpha=0.8
    )

    # Add decision hyperplane
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 10),
        np.linspace(-1, 1, 10)
    )
    z = (-mlp.W2[0, 0] * xx - mlp.W2[1, 0] * yy - mlp.b2[0, 0]) / mlp.W2[2, 0]
    ax_hidden.plot_surface(xx, yy, z, alpha=0.3, color="gray")

    ax_hidden.set_xlim([-1, 1])
    ax_hidden.set_ylim([-1, 1])
    ax_hidden.set_zlim([-1, 1])
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")

    # === Input Space Visualization ===
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)  # Forward pass for the decision boundary
    ax_input.clear()
    ax_input.contourf(xx, yy, preds, cmap="bwr", alpha=0.8)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    # === Gradient Visualization ===
    ax_gradient.clear()
    grad_magnitudes_W1 = np.linalg.norm(mlp.W1, axis=0)
    grad_magnitudes_W2 = np.abs(mlp.W2).flatten()
    grad_magnitudes_W1 /= grad_magnitudes_W1.max()  # Normalize for visualization
    grad_magnitudes_W2 /= grad_magnitudes_W2.max()

    # Input Nodes
    for i in range(mlp.W1.shape[0]):
        ax_gradient.scatter(i, 0, c="blue", s=500)
        ax_gradient.text(i, -0.2, f"x{i+1}", ha="center", va="top")

    # Hidden Nodes
    for j in range(mlp.W1.shape[1]):
        ax_gradient.scatter(j + 2, 1, c="green", s=500)
        ax_gradient.text(j + 2, 1.2, f"h{j+1}", ha="center", va="bottom")

    # Output Node
    ax_gradient.scatter(mlp.W1.shape[1] + 2, 2, c="red", s=600)
    ax_gradient.text(mlp.W1.shape[1] + 2, 2.2, "y", ha="center", va="bottom")

    # Connections
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            ax_gradient.plot([i, j + 2], [0, 1], color="purple", alpha=grad_magnitudes_W1[j])
    for j in range(mlp.W1.shape[1]):
        ax_gradient.plot([j + 2, mlp.W1.shape[1] + 2], [1, 2], color="orange", alpha=grad_magnitudes_W2[j])

    ax_gradient.set_xlim([-1, mlp.W1.shape[1] + 3])
    ax_gradient.set_ylim([-1, 3])
    ax_gradient.axis("off")
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")



def visualize(activation, lr, step_num):
    try:
        X, y = generate_data()
        mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
        fig = plt.figure(figsize=(21, 7))
        ax_hidden = fig.add_subplot(131, projection='3d')
        ax_input = fig.add_subplot(132)
        ax_gradient = fig.add_subplot(133)

        # Set up visualization
        fig = plt.figure(figsize=(21, 7))  # Create the figure here
        ax_hidden = fig.add_subplot(131, projection='3d')  # Hidden space (3D if 3 hidden nodes)
        ax_input = fig.add_subplot(132)  # Input space
        ax_gradient = fig.add_subplot(133)  # Gradient visualization

        # Create animation
        ani = FuncAnimation(
            fig,
            update,
            fargs=(mlp, ax_input, ax_hidden, ax_gradient, X, y),
            frames=step_num // 10,
            repeat=False
        )

        # Save the animation as a GIF
        ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=30)  
        plt.close(fig)  # Close the figure after saving the animation
        return True
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return False



if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
