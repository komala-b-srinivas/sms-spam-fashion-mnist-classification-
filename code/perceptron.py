import numpy as np

class Perceptron:

    

    def __init__(self, max_iters=1000):

        self.max_iters = max_iters

        # parameters
        self.w = None
        self.b = None

        # stores number of iterations training took to converge
        self.num_iterations = None

    def train(self, X, y):


        w = np.zeros(X.shape[1])
        b = 0
        num_iterations = 0

        for it in range(self.max_iters):
            errors = 0
            for xi, target in zip(X, y):
                z = np.dot(w, xi) + b
                if target * z <= 0:
                    w = w + target * xi
                    b = b + target
                    errors += 1
            num_iterations += 1
            if errors == 0:
                break

        self.w = w
        self.b = b
        self.num_iterations = num_iterations

    def predict(self, X):


        y_pred = np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)
        return y_pred

    def name(self):
        """
        Name of model
        """
        return 'Perceptron'
