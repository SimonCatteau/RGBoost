import numpy as np


def execute_split(feature_data, index_list, best_split):
    """
    Executes a split
        - updates the index lists with the data that is still covered by the current formed rule
    :param feature_data:
    :param index_list:
    :param best_split:
    :return:
    """
    # initialization
    split_feature = best_split.feature
    split_type = best_split.type
    split_value = best_split.value
    new_index_list = []
    # check which data passes the new split
    for (i, index) in enumerate(index_list):
        if (split_type == "lt" and feature_data[index][split_feature] <= split_value) or (split_type == "gt" and feature_data[index][split_feature] > split_value):
            new_index_list.append(index)
    return new_index_list

