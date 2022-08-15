import numpy as np
def find_best_split(best_split,prev_gain, feature_data, nb_bins, index_list, g_vector, gain_treshold):
    """
    finds the best split for given data
    :param best_split: previous best split
    :param prev_gain: previous gain
    :param feature_data: all the trainingdata
    :param nb_bins: number of buckets
    :param index_list: indices of the trainin examples that are still covered by the rule
    :param g_vector: the gradients
    :param gain_treshold: minimal relative gain of adding a split
    :return: best split, and a bool indicating if the rule learning should stop
    """
    # stop criteria
    stop = True
    active_feature_data = np.take(feature_data, index_list, axis=0)
    active_g_vector = np.take(g_vector, index_list, axis=0)
    nb_features = len(feature_data[0])

    for feature in range(nb_features):

        # binning the feature_vector
        feature_vector = active_feature_data[:, feature]  # this selects the column (this is numpy syntax)
        min_value, max_value = np.min(feature_vector), np.max(feature_vector)
        buckets = np.linspace(min_value, max_value, nb_bins)
        binned_data = np.digitize(feature_vector, buckets, right=True)
        bin_count = np.bincount(binned_data)  # bin_count[0] are the number of examples that are in bin 1
        buckets, binned, bin_count = remove_empty_buckets(buckets, binned_data, bin_count)

        # bin the g's according to the binning of the feature vector
        sum_of_binned_g = np.zeros(len(bin_count))
        for b in range(len(bin_count)):
            sum_of_binned_g[b] = np.sum(active_g_vector[binned_data == b])

        # loop over the bins and select the best split
        for b in range(1, len(bin_count)):  # splitting on bin 0, is should make a split on list after position 1
            bin_count_left = np.sum(bin_count[:b])
            bin_count_right = np.sum(bin_count[b:])
            g_sum_left = np.sum(sum_of_binned_g[:b])
            g_sum_right = np.sum(sum_of_binned_g[b:])
            left_split = (g_sum_left**2)/bin_count_left  # this equals sum(G)²/sum(H)
            right_split = (g_sum_right**2)/bin_count_right
            if right_split >= left_split and right_split > best_split.gain and check_if_better_split(prev_gain, right_split,gain_treshold):
                best_split = Split(feature, "gt", buckets[b - 1], right_split)
                stop = False
            elif left_split > best_split.gain and check_if_better_split(prev_gain, left_split, gain_treshold):  # gt is the best split up until now
                best_split = Split(feature, "lt", buckets[b - 1], left_split)
                stop = False
    return best_split, stop


def remove_empty_buckets(buckets, binned_data, bin_count):
    """
    removes empty buckets from the buckets. And updates the others boundries
    in this way beter split candidates are formed
    :param buckets:
    :param binned_data:
    :param bin_count:
    :return:
    """
    buckets_width = buckets[1] - buckets[0]
    buckets_center = buckets_width/2
    index = 0
    deleted_items = 0
    length = len(bin_count)
    while index + deleted_items <= length-1:
        if bin_count[index] == 0:
            buckets[index - 1] += buckets_center
            buckets = np.delete(buckets, index)
            bin_count = np.delete(bin_count, index)
            for bin in range(len(binned_data)):
                if binned_data[bin] >= index:
                    binned_data[bin] -= 1
            deleted_items += 1
        else:
            index += 1
    return buckets, binned_data, bin_count


class Split:
    """
    object that stores the information of a split
    """
    def __init__(self, feature, type, value, gain):
        self.feature = feature  # a number ranging from 0 to nb_features -1
        self.type = type  # "gt" for greater than and "lt" for less than
        self.value = value  # numerical value of the split
        self.gain = gain  # the loss associated with the split


    def __str__(self):
        return "Split on feature {} {} {} with a gain of {}".format(self.feature, self.type, self.value, self.gain)

def check_if_better_split(prev_gain, new_gain, gain_treshold):
    """
    insures that the gain treshold is reached
    :param prev_gain:
    :param new_gain:
    :param gain_treshold:
    :return:
    """
    if prev_gain is None:
        return True
    else:
        return (new_gain - prev_gain) > (prev_gain*gain_treshold)






