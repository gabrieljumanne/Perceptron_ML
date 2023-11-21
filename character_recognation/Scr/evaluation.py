# code for the model evaluation

"""
    It helps to understand how well our  model performing is , identifying the strength, and weakness
    and guiding improvement .
"""

def accuracy(y_true, y_pred):
    """
    calculate the accuracy of  the model


    :param y_true:(numpy.ndarray)- True labels
    :param y_pred:(numpy.ndarray) - predicted values
    :return:-float- Accuracy score
    """
    # calculate the number of correct prediction(total number of true values )
    correct_prediction = sum(y_true == y_pred)
    # calculate the accuracy
    accu_racy = correct_prediction / len(y_true)

    return accu_racy

def precision(y_true, y_pred):
    """
    calculate the precision of the model .

    :param y_true:(numpy.ndarray) - True labels
    :param y_pred:(numpy.ndarray) - Predicted labels
    :return:-float- Precision score
    """
    # calculate True Positive (TP) and False Positive (FP) in our model
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))

    # Avoid division by zero -- handling the zero
    if tp + fp == 0:
        return 0
    # calculate the recall
    recall = tp / (tp + fp)

    return recall


