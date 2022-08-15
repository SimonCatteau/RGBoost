import numpy as np


def execute_rule(rule, feature_data, target_data, prediction_update, g_mean):
    """
    executes the boosting updates for a given rule
        - update the predictions
        - update the gradients
    :param rule:
    :param feature_data:
    :param target_data:
    :param prediction_update:
    :param g_mean:
    :return:
    """
    # updating the prediction
    for example_nb, example_features in enumerate(feature_data):
        triggered_by_rule = True
        for index in range(len(rule)):
            split = rule[index]
            if example_features[split.feature] > split.value and split.type == "lt":
                triggered_by_rule = False
            elif example_features[split.feature] <= split.value and split.type == "gt":
                triggered_by_rule = False
        if triggered_by_rule:
            prediction_update[example_nb] += g_mean
    # updating the g vector
    g_vector_update = np.subtract(target_data, prediction_update)
    return prediction_update, g_vector_update
