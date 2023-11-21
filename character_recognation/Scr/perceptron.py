# class for the perceptron
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Initialize the perceptron model( the perceptron classifier)


        :param learning_rate: the learning rate for the weight update
        :param epochs: the number of training epoch
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def initialize_weights(self, num_features):
        """
     This initializes  the weight according to  for the number of features

     :param num_features: The number of features in the input data
     :return:
     """
        # initialize the weight with small random numbers
        self.weights = np.random.rand(num_features)
        # initialize bias to to zero
        self.bias = 0.0

    def predict(self, x):
        """
        Make the prediction using the current model.

        :param x:   (numpy.ndarray) input features
        :return: (numpy.ndarray) predicted labels (1, 0 ) ---this is one of the chosen array
        """
        # calculate the weight of sum of the input
        weighted_sum = np.dot(x, self.weights) + self.bias

        # apply the step function (weighted_sum) to get binary prediction (0 or 1 )
        prediction = np.where(weighted_sum >= 0, 1, 0)
        return prediction

    def train(self, x_train, y_train):
        """
        Train the perceptron model

        :param x_train:(numpy.ndarray) Training features
        :param y_train:(numpy.ndarray) Training labels
        :return: No any return value ..
        """

        # the shape[ 1] takes the second index of the tuple which is the number of feature dimension
        num_features = x_train.shape[1]
        self.initialize_weights(num_features)

        for epoch in range(self.epochs):
            for x, y in zip(x_train, y_train):

                # make the prediction
                prediction = self.predict(x)
                # update the weight if only misclassficastion  occurs
                if prediction != y:
                    # update the weight
                    self.weights += self.learning_rate * (y - prediction)*x
                    # update the bias
                    self.bias += self.learning_rate * (y-prediction)











