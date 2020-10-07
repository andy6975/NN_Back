import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2868)


class MNIST:

    @classmethod
    def load_data(cls):
        num_classes = 10

        train, test = tf.keras.datasets.mnist.load_data()
        
        X_train, y_train = train[0], train[1]
        X_test, y_test = test[0], test[1]
        
        y_train = np.array(y_train)
        y_one_hot_train = np.zeros((y_train.size, y_train.max()+1))
        y_one_hot_train[np.arange(y_train.size), y_train] = 1

        y_test = np.array(y_test)
        y_one_hot_test = np.zeros((y_test.size, y_test.max() + 1))
        y_one_hot_test[np.arange(y_test.size), y_test] = 1

        return X_train, y_one_hot_train, X_test, y_one_hot_test

class FC_Layer:

    def __init__(self,
                num_inputs,
                layer_size, 
                d_activation_fn,
                activation_fn):
        super().__init__()

        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn
        
        self.d_activation_fn = d_activation_fn
        self.x, self.y, self.dL_dW, self.dL_db = 0, 0, 0, 0

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        self.y = self.activation_fn(z)
        self.x = x
        return self.y

    def backward(self, dL_dy):

        dy_dz = self.d_activation_fn(self.y)
        dL_dz = (dL_dy * dy_dz)
        dz_dw = self.x.T
        dz_dx = self.W.T
        dz_db = np.ones(dL_dy.shape[0])

        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        dL_dx = np.dot(dL_dz, dz_dx)

        return dL_dx

    def optimize(self, epsilon):
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db

class FC_Network(FC_Layer):

    def __init__(self,
                num_inputs,
                num_outputs,
                hidden_layer_sizes=[64, 32]):
        sizes = [num_inputs, *hidden_layer_sizes, num_outputs]
        self.layers = [FC_Layer(sizes[i], sizes[i+1], self.derivated_sigmoid, self.sigmoid)
                        for i in range(len(sizes) - 1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivated_sigmoid(self, x):
        return x * (1 - x)

    def loss_L2(self, pred, target):
        return np.sum(np.square(pred - target)) / pred.shape[0]

    def derivated_loss_L2(self, pred, target):
        return 2 * (pred - target)

    def forward_pass(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward_pass(self, dL_dy):
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)

        return dL_dy

    def optimize_network(self, epsilon):
        for layer in self.layers:
            layer.optimize(epsilon)

    def predict(self, x):
        estimation = self.forward_pass(x)
        best_classes = np.argmax(estimation)

        return best_classes

    def evaluate_accuracy(self, X_val, y_val):
        num_corrects = 0
        for i in range(len(X_val)):
            if self.predict(X_val[i]) == np.argmax(y_val[i]):
                num_corrects += 1

        return num_corrects / len(X_val)
    
    def train(self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
            num_epochs=10,
            learning_rate=1e-3):

            num_batches_per_epoch = X_train.shape[0] // batch_size
            loss, accuracy = [], []

            for i in range(num_epochs):
                epoch_loss = 0

                for b in range(num_batches_per_epoch):
                    b_idx = b * batch_size
                    b_idx_e = b_idx + batch_size
                    x, y_true = X_train[b_idx:b_idx_e], y_train[b_idx:b_idx_e]

                    y = self.forward_pass(x)
                    epoch_loss += self.loss_L2(y, y_true)
                    dL_dy = self.derivated_loss_L2(y, y_true)
                    self.backward_pass(dL_dy)
                    self.optimize_network(learning_rate)

                loss.append(epoch_loss / num_batches_per_epoch)
                accuracy.append(self.evaluate_accuracy(X_val, y_val))
                print("Epoch: {:4d}    training_loss: {:.6f}    Val_Accuracy: {:.2f}%".format(i+1, loss[i], accuracy[i] * 100))

X_train, y_train, X_test, y_test = MNIST.load_data()
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

mnist_classifier = FC_Network(784, 10, [64, 32, 16])
mnist_classifier.train(X_train, y_train, X_test, y_test, num_epochs=500, batch_size=64)