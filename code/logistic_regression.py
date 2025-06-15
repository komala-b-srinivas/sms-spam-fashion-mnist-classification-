import numpy as np

class LogisticRegression:
    

    def __init__(self, max_iters=1000, alpha=.05):

        self.alpha = alpha
        self.max_iters = max_iters

        # parameters
        self.w = None
        self.b = None

        # stores training loss per each iteration (epoch)
        self.training_loss_per_epoch = []

    def sigmoid(self, x):

        sig = 1 / (1 + np.exp(-x))
        return sig

    def log_loss(self, y, y_hat, epsilon=1e-16):

        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        log_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return log_loss

    def train(self, X, y):


        w = np.zeros(X.shape[1])
        b = 0
        training_loss = []

        for _ in range(self.max_iters):
            w_prev = w.copy()
            b_prev = b

            for xi, yi in zip(X, y):
                z = np.dot(w, xi) + b
                a = self.sigmoid(z)
                dz = a - yi
                dw = dz * xi
                db = dz
                w -= self.alpha * dw
                b -= self.alpha * db

            y_hat = self.sigmoid(np.dot(X, w) + b)
            loss = self.log_loss(y, y_hat)

            if len(training_loss) > 0 and loss >= training_loss[-1]:
                w = w_prev
                b = b_prev
                break

            training_loss.append(loss)

        self.w = w
        self.b = b
        self.training_loss_per_epoch = training_loss

    def predict(self, X, threshold=.5):

        y_hat = self.sigmoid(np.dot(X, self.w) + self.b)
        y_pred = (y_hat > threshold).astype(int)
        return y_pred

    def name(self):
        """
        Name of model
        """
        return 'Logistic Regression'
