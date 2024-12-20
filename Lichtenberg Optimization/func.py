from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class ObjectiveFunction(ABC):
    @abstractmethod
    def bounds(self):
        """Return two floats representing the bounds of the input space."""
        pass
    
    @abstractmethod
    def center(self):
        """Return the center point of the bounded function input space."""
        pass
    
    @abstractmethod
    def evaluate(self, *args):
        """Evaluate the objective function for the provided arguments."""
        pass


class AckleyFunction(ObjectiveFunction):
    def __init__(self, dimensions=3):
        """Initialize the dimensions and bounds for evaluating the Ackley function."""
        self.dim = dimensions - 1
        self.lower_bound = -32.768
        self.upper_bound = 32.768
    
    def bounds(self):
        """Return the bounds of the search space as a tuple of two floats."""
        return self.lower_bound, self.upper_bound
    
    def center(self):
        """Return the center of the bounded search space for the given dimension as a tuple."""
        mean = (self.lower_bound + self.upper_bound) / 2
        return tuple(mean for _ in range(self.dim))
    
    def evaluate(self, *args):
        """
        Evaluate the Ackley function for n-dimensional inputs.
        
        Args:
            args (list): a list of input values with length equal to the number of dimensions.

        Returns:
            int: the output of the Ackley function given the args.    
        
        Raises:
            ValueError: caused by a mismatch between given and expected dimensions.
        """
        if len(args) != self.dim:
            raise ValueError(f"Expected {self.dim} arguments, got {len(args)}.")
        
        x = np.array(args)
        
        # Parameters of the Ackley function
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        # Ackley function formula
        sum_sq_term = -b * np.sqrt(np.mean(x ** 2))
        cos_term = np.mean(np.cos(c * x))
        result = -a * np.exp(sum_sq_term) - np.exp(cos_term) + a + np.exp(1)
        
        return result
    
    def plot3d(self, *points):
        """
        Plot the Ackley function in 3D for visualization.
        """
        if self.dim != 2:
            raise ValueError("Plotting is only supported for 3D functions.")

        x = np.linspace(self.lower_bound, self.upper_bound, 50)
        y = np.linspace(self.lower_bound, self.upper_bound, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.array([self.evaluate(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', alpha=0.7)

        if points:
            colors = ['red', 'green', 'blue']
            for i, point in enumerate(points):
                ax.scatter(point[0], point[1], self.evaluate(*point), s=50, alpha=1, color=colors[i%3], label=f'$X_{i}$')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

class BoothFunction(ObjectiveFunction):
    def __init__(self):
        """Initialize the dimensions and bounds for evaluating the Booth function."""
        self.dim = 2  # The Booth function is defined in a 2D space
        self.lower_bound = -10
        self.upper_bound = 10
    
    def bounds(self):
        """Return the bounds of the search space as a tuple of two floats."""
        return self.lower_bound, self.upper_bound
    
    def center(self):
        """Return the center of the bounded search space for the given dimension."""
        mean = (self.lower_bound + self.upper_bound) / 2
        return tuple(mean for _ in range(self.dim))
    
    def evaluate(self, *args):
        """
        Evaluate the Booth function for 2-dimensional inputs.
        
        Args:
            args (list): a list of input values of length 2.

        Returns:
            int: the output of the Booth function given the args.    
        
        Raises:
            ValueError: raised when list of args is not 2-dimensional.
        """
        if len(args) != self.dim:
            raise ValueError(f"Expected {self.dim} arguments, got {len(args)}.")
        
        x1, x2 = args
        
        # Booth function formula
        result = (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
        
        return result
    
    def plot3d(self, *points):
        """
        Plot the Booth function in 3D for visualization.
        """
        if self.dim != 2:
            raise ValueError("Plotting is only supported for 2D functions.")

        x = np.linspace(self.lower_bound, self.upper_bound, 50)
        y = np.linspace(self.lower_bound, self.upper_bound, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.array([self.evaluate(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', alpha=0.7)

        if points:
            colors = ['red', 'green', 'blue']
            for i, point in enumerate(points):
                ax.scatter(point[0], point[1], self.evaluate(*point), s=50, alpha=1, color=colors[i%3], label=f'$X_{i}$')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Function Value')
        ax.set_title('Booth Function 3D Plot')
        plt.show()

