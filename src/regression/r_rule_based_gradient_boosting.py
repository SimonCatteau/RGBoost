import copy
from src.regression.r_split_finder import find_best_split, Split
from src.regression.r_split_executer import execute_split
from src.regression.r_rule_executer import execute_rule
from src.regression.r_model import Model
from math import floor
import numpy as np
from random import sample
from sklearn.metrics import mean_squared_error


def RGBoostRegressor(feature_data, target_data, iterations=1000, learning_rate=0.01, sample_size=1, max_depth=5,
                     coarsity=3, min_loss=0, nb_buckets=10, gain_treshold=0):
    """
    This is the src regressor main algorithm
    :param feature_data: the training data
    :param target_data: the target data
    :param iterations: number of boosting iterations (=rules to learn)
    :param learning_rate: scaling of the update
    :param sample_size: relative sample size
    :param max_depth: max length of rule
    :param coarsity: number of examples that a rule should cover
    :param min_loss: minimal loss on the trainindata
    :param nb_buckets: number of buckets used in the split finding algorithm
    :param gain_treshold:
    :return: a model that captures the rule and the training data
    """

    # initialization
    train_loss = np.array([])
    nb_examples = len(target_data)
    original_index_list = range(nb_examples)
    target_mean = np.mean(target_data)
    target_prediction = np.repeat(target_mean, nb_examples)
    g_vector = np.subtract(target_data, target_prediction)
    loss = mean_squared_error(target_data, target_prediction)
    learned_model = Model(target_mean)
    m = 0

    # gradient boosting loop
    while not stop_boosting(iterations, min_loss, m, loss):
        index_list = sample(original_index_list, floor(len(original_index_list)*sample_size))
        while_iteration = 1
        best_rule = np.array([])
        shrinking_index_list = copy.deepcopy(index_list)
        best_split = Split(None, None, None, 0)
        stop = False

        # rule learning loop
        while not stop_rule_learning(stop, while_iteration, shrinking_index_list, coarsity, max_depth):
            prev_gain = best_split.gain
            best_split, stop = find_best_split(best_split, prev_gain, feature_data, nb_buckets, shrinking_index_list, g_vector, gain_treshold)
            if not stop:
                best_rule = np.append(best_rule, best_split)
                shrinking_index_list = execute_split(feature_data, shrinking_index_list, best_split)
                while_iteration += 1

        # updating the prediction after adding the rule learned in the while loop
        g_mean_update = np.mean(np.take(g_vector, shrinking_index_list, axis=0))*learning_rate
        learned_model.add_rule(best_rule, g_mean_update)
        # the last split its g_mean is the g_mean of the remaining examples
        target_prediction, g_vector = execute_rule(best_rule, feature_data, target_data, target_prediction, g_mean_update)
        loss = mean_squared_error(target_data, target_prediction)
        m += 1
        train_loss = np.append(train_loss, loss)
        if m % 250 == 0:
            print("ITERATION {}: loss is {}".format(m, loss))
    return learned_model, train_loss


def stop_rule_learning(stop, while_iterations, feature_data, coarsity, max_depth):
    """
    checks if the rule learning proces should stop
    :param stop: true if no suited split was found
    :param while_iterations: the length of the current rule
    :param feature_data: current number of examples that the rule covers
    :param coarsity: minimal number of examples that should be covered by a rule
    :param max_depth: max length of the rule
    :return: bool
    """
    return stop or while_iterations -1 == max_depth or len(feature_data) <= coarsity


def stop_boosting(iterations, min_loss, m, loss):
    """
    check if algo should stop
    :param iterations: max iterations
    :param min_loss: min loss
    :param m: current iteration
    :param loss: current loss
    :return: bool
    """
    return m >= iterations or loss <= min_loss

