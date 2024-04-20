
import numpy as np
import matplotlib as mlp
import csv, math

class LogisticRegression:
    def forward_prop(self, x, w):
        return self.activation(np.matmul(x, w))
    
    """ sigmoid activatoin function"""
    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    """ MSE loss function (guess & label are arrays)"""
    def MSE_loss(self, guess, label):
        return np.mean(.5 * ((guess - label) ** 2))
    
    ### This shit is fucked vvvvvvvvv
    def loss(self, y_hat, y):
        sum = 0.0
        for i in range(len(y)):
            y1 = y_hat[i]
            y2 = y[i]

            sum += -(y2*math.log(y1) + (1 - y2) * math.log(1 - y1))

        return sum


    
    """ Stochastic Gradient Decent with Mini-Batches 
        Features and Labels should be from the training set """
    def fit(self, features, labels, max_epochs=100, learning_rate=.01, batch_size=32):

        N = len(features)
        d = len(features[0])
        w = np.zeros(d, dtype=float)

        num_batches_per_epoch = int(N / batch_size)

        for i in range(max_epochs):

            y = self.forward_prop(features, w)
            loss = self.MSE_loss(y, labels)
            print("Epoch:", i + 1, "\tLoss:", loss)


            for i in range(len(features)):
                x = features[i]
                y = labels[i]

                y_hat = self.forward_prop(x, w)

                gradient = y_hat - y

                w -= learning_rate * (x * gradient)


"""
            for j in range(num_batches_per_epoch):
                # Create Random Batches
                indeces = np.random.choice(features.shape[0], 4, replace=False)

                X = np.array(features[indeces], dtype=float)
                Y = np.array(labels[indeces], dtype=float)

                # gradient calculations
                batch_y_hat = self.forward_prop(X, self.w)
                mean_loss = batch_y_hat - Y

                #print(batch_y_hat.shape, mean_loss.shape)

                self.w -= (2 * learning_rate * np.sum(np.dot(X.transpose(), mean_loss)) / batch_size)

                #gradient = np.dot(batch_y_hat.transpose(), mean_loss)
                #self.w = self.w - (learning_rate * gradient)
"""

df_training = open("mnist_train.csv")
df_testing = open("mnist_test.csv")

# ignore column names lines
df_testing.readline()
df_training.readline()

buffer = [[float(x) for x in line.split(',')] for line in df_training.readlines()]

training_labels = np.array([data[0] for data in buffer], dtype=np.double)
training_features = np.array([data[1:] for data in buffer], dtype=np.double)

buffer = [[float(x) for x in line.split(',')] for line in df_testing.readlines()]

testing_labels = np.array([data[0] for data in buffer], dtype=np.double)
testing_features = np.array([data[1:] for data in buffer], dtype=np.double)

zero_testing_labels = np.array([int(x == 0) for x in testing_labels], dtype=np.double)
zero_training_labels = np.array([int(x == 0) for x in testing_labels], dtype=np.double)

#for label in training_labels:
#    np.append(label, 1)

model = LogisticRegression()
model.fit(testing_features, zero_testing_labels, max_epochs=1000)
