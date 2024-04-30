import numpy as np

class LogisticRegression:

    """ Forward Propagation """
    def forward_prop(self, x, w):
        return self.activation(np.matmul(x, w))
    
    """ sigmoid activation function"""
    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    """ MSE loss function (guess & label are np.arrays)"""
    def MSE_loss(self, guess, label, w):
        return (np.sum((np.square(np.subtract(guess, label)) / 2)) / len(label)) + np.sum(np.square(w))

    """ Returns the error rate, expressed as a percentage, of the given dataset """
    def error_rate(self, features, labels, w):

        if len(features) != len(labels):
            print("invalid data")
            return

        errors = 0.0
        y_hat = self.forward_prop(features, w)

        for i in range(len(labels)):
            guess = round(y_hat[i])
            if guess != labels[i]:
                errors += 1

        return errors / len(labels)

    
    #   Gradient Decent Implementation
    #   Features and Labels should be from the training set
    def fit(self, training_features, training_labels, max_epochs=100, learning_rate=.01):

        if len(training_features) != len(training_labels):
            print("invalid data")
            return
        
        # initialize weights
        d = len(training_features[0])
        w = np.random.normal(0.5, 0.2, size=(d,))

        for i in range(max_epochs):
            
            # every 10 epochs
            if i % 10 == 0:
                y = self.forward_prop(training_features, w)
                loss = self.MSE_loss(y, training_labels, w)
                print("Epoch:", i, "\tLoss:", loss)

                # learning rate step decay
                learning_rate *= .95

            # calculate prediction vector for entire dataset
            y_hat = self.forward_prop(training_features, w)

            # compute gradient with added weight constraints
            gradient = np.matmul(np.subtract(y_hat, training_labels), training_features)

            # weight update
            w -= learning_rate * gradient

            # weight constraint
            w = w / np.sqrt(np.sum(w**2))

        return w