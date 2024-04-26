
import numpy as np
import math

class LogisticRegression:

    """ Forward Propagation """
    def forward_prop(self, x):
        if self.w is not None:
            return self.activation(np.matmul(x, self.w))
        return 0
    
    """ sigmoid activation function"""
    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    """ MSE loss function (guess & label are np.arrays)"""
    def MSE_loss(self, guess, label):
        return np.mean(.5 * ((guess - label) ** 2))
    
    """ cross entropy loss function 
        (broken atm)"""
    def loss(self, y_hat, y):
        sum = 0.0
        for i in range(len(y)):
            y1 = y_hat[i]
            y2 = y[i]
            sum += -(y2*math.log(y1) + (1 - y2) * math.log(1 - y1))

        return sum

    """ Returns the error rate, expressed as a percentage, of the given dataset """
    def error_rate(self, features, labels):

        if len(features) != len(labels):
            print("invalid data")
            return

        errors = 0.0
        y_hat = self.forward_prop(features)

        for i in range(len(labels)):
            guess = 0 if y_hat[i] < .5 else 1
            if guess != labels[i]:
                errors += 1

        return errors / len(labels)

    
    """ Stochastic Gradient Decent with Mini-Batches 
        Features and Labels should be from the training set """
    def fit(self, features, labels, max_epochs=100, learning_rate=.01, max_weight_normal=1):

        if len(features) != len(labels):
            print("invalid data")
            return

        N = len(features)
        d = len(features[0])
        self.w = np.zeros(d, dtype=float)

        for i in range(max_epochs):

            if not i % 10:
                y = self.forward_prop(features)
                loss = self.MSE_loss(y, labels)
                print("Epoch:", i, "\tLoss:", loss)

            y_hat = self.forward_prop(features)
            gradient = np.subtract(y_hat, labels)

            self.w -= learning_rate * (np.matmul(gradient, features))
            
            """ Weight Constraints """
            normal = np.linalg.norm(self.w)
            if normal > max_weight_normal:
                self.w *= max_weight_normal / normal



df_training = open("mnist_train.csv")
df_testing = open("mnist_test.csv")

# ignore column names lines
df_testing.readline()
df_training.readline()

buffer = [[float(x) for x in line.split(',')] for line in df_training.readlines()]

training_labels = np.array([data[0] for data in buffer], dtype=np.int_)
training_features = np.array([data[1:] for data in buffer], dtype=np.double)
training_features /= 255

buffer = [[float(x) for x in line.split(',')] for line in df_testing.readlines()]

testing_labels = np.array([data[0] for data in buffer], dtype=np.double)
testing_features = np.array([data[1:] for data in buffer], dtype=np.double)
testing_features /= 255

model = LogisticRegression()

# Fit model for all 10 digits
for d in range(3, 9):
    new_testing_labels = np.array([int(x == d) for x in testing_labels], dtype=np.int_)
    new_training_labels = np.array([int(x == d) for x in training_labels], dtype=np.int_)

    model.fit(training_features, new_training_labels, max_epochs=100, learning_rate=0.001)
    print("Digit:", d, "\tErorr Rate:", model.error_rate(testing_features, new_testing_labels))

    # save weights to file
    #filename = "weights/w" + str(d) + "_normalized_weights.txt"
    #model.w.tofile(filename)