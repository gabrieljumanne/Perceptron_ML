"""
    Perceptron logic class from the logic of artificial neuron
    here i aam building the artificial neurons .
"""
# implementation of peceptron in python
import numpy as np

# ===================================================================================
"""
    perceptron learning model: This class  represent the learning model for the 
        perceptron using python ....
"""
# ===================================================================================
# object is the base class in python mostly used in python 2

class Perceptron:
    """
    perceptron classifier -the blueprint

    parameters/ characteristics
    ----------------------------
    eta : float
        learning rate (between 0.0 and 1.0) .

    n_iter : int
         passes over the training data set ( epoch) ... this is number of iteration .

    random_state : int.
        Random number generator seed for the random weight
        initialization

    attributes/ variables
    --------------------

    w_ : 1d-array
         weight after fitting for the update .
    errors_ : list
        Number of mis classification (updates ) in each epoch (n_iter).

    """
    # the initialization method  for my class( blue-print)
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # class attributes

    def fit(self, x, y):
        """
        Fit training data. - for training the simple data  .
        ------------------
        X: { Array alike }, shape = [n_examples, n_features]
            Training vector,  where n_example is the numbers of examples ,
            n_features is the number of features .
            - x: is for training examples

        y: {Array alike } , shape = [n_examples]
            for Target values .

        Returns
        ---------
        self: object
        """

        # RandomState is the class provided by the numpy to creates generators objects with specified seed
        rgen = np.random.RandomState(self.random_state)

        # Weight initialize including one for the bias
        # Generating the initial weight
        # This ensures the weight vector,  matches to the number of features

        # This is weight vector as :
        # generates the weights according to the number of features .
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])



        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                # for each element x, y
                update = self.eta * (target - self.predict(xi))  # update mechanism
                self.w_[1:] += update * xi  # the update of the weight from index 1
                self.w_[0] += update        # the update of the weight for the bias element
                errors += int(update != 0.0)     # When update is not equal to 0.0 should update
            self.errors_.append(errors)

            return self

    # this creates the net input of features and there weight
    def net_input(self, x):
        """ Calculate the net input """
        # self.w_[0] is the bias value .
        # self.w_[1:] its takes all the value till the ....n_numbers
        # the bias element is very important for prediction
        return np.dot(x, self.w_[1:] + self.w_[0])

    def predict(self, x):
        """ return class label after the unit step """
        return np.where(self.net_input(x) >= 0.0, 1, -1)




