import copy
from src.classification.c_split_finder import find_best_split, Split
from src.classification.c_split_executer import execute_split
from src.classification.c_rule_executer import execute_rule, update_g_vector
from src.classification.c_model import Model
from math import floor
import numpy as np
from random import sample
from sklearn.metrics import log_loss


def RGBoostClassifier(feature_data, target_data, iterations=1000, learning_rate=0.01, sample_size=1, max_depth=3,
                      coarsity=3, min_loss=0, nb_buckets=10, gain_treshold=0):
    """
    This is the src classifier main algorithm
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
    train_accuracy = np.array([])
    nb_examples = len(target_data)
    original_index_list = range(nb_examples)
    nb_neg = target_data[np.where(target_data < 0.5)].shape[0]
    nb_pos = nb_examples - nb_neg
    target_mean = (-1 * nb_neg + 1 * nb_pos)/nb_examples
    prior = 0.5 * np.log((1 + target_mean) / (1 - target_mean))
    target_prediction = np.repeat(prior, nb_examples)
    vectorized_update = np.vectorize(update_g_vector)
    g_vector = vectorized_update(target_data, target_prediction)
    loss = log_loss(target_data, target_prediction)
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
        while not stop_rule_learning(stop, while_iteration, shrinking_index_list, max_depth, coarsity):
            prev_gain = best_split.gain
            best_split, stop = find_best_split(best_split, prev_gain, feature_data, nb_buckets, shrinking_index_list, g_vector, gain_treshold)
            if not stop:
                best_rule = np.append(best_rule, best_split)
                shrinking_index_list = execute_split(feature_data, shrinking_index_list, best_split)
            while_iteration += 1

        # updating the prediction after adding the rule learned in the while loop
        g_mean_update = calculate_g_update(shrinking_index_list, g_vector)*learning_rate
        learned_model.add_rule(best_rule, g_mean_update)
        # the last split its g_mean is the g_mean of the remaining examples
        target_prediction, g_vector = execute_rule(best_rule, feature_data, target_data, target_prediction, g_mean_update)
        apply_loss = np.vectorize(binary_log_loss)
        loss = np.sum(apply_loss(target_data, target_prediction))/len(target_data)
        m += 1
        apply_prediction = np.vectorize(get_class_label_for_example)
        predicion_0_1 = apply_prediction(target_prediction)
        accuracy = get_accuracy(target_data, predicion_0_1)
        train_accuracy = np.append(train_accuracy, accuracy)
        if m % 250 == 0:
            print("ITERATION {}: loss is {}".format(m, loss))
    return learned_model, train_accuracy


def stop_rule_learning(stop, while_iterations, feature_data, max_depth, coarsity):
    """
    checks if the rule learning proces should stop
    :param stop: true if no suited split was found
    :param while_iterations: the length of the current rule
    :param feature_data: current number of examples that the rule covers
    :param coarsity: minimal number of examples that should be covered by a rule
    :param max_depth: max length of the rule
    :return: bool
    """
    return stop or while_iterations-1 == max_depth or len(feature_data) <= coarsity


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


def calculate_g_update(shrinking_index_list, g_vector):
    """
    update g according to Friedman paper 2008
    :param shrinking_index_list:
    :param g_vector:
    :return:
    """
    g_list = -np.take(g_vector, shrinking_index_list)
    g_list_absolute = np.absolute(g_list)
    return np.sum(g_list)/np.sum(g_list_absolute*(2-g_list_absolute))


def binary_log_loss(target_data, target_prediction):
    """
    calculates log loss, makes sure that values are between -1 en 1
    :param target_data:
    :param target_prediction:
    :return:
    """
    target_data = 2 * target_data - 1
    return np.log(1 + np.exp(-2 * target_data * target_prediction))


def get_class_label_for_example(prediction):
    """
    get classlabel
    :param prediction:
    :return:
    """
    if prediction > 0:
        label = 1
    else:
        label = 0
    return label

def get_accuracy(y_test, y_predict):
    """
    get accuracy
    :param y_test:
    :param y_predict:
    :return:
    """
    return 1 - (np.sum(np.absolute(np.subtract(y_test, y_predict))) / y_test.shape[0])