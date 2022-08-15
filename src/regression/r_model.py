import numpy as np
from sklearn.metrics import mean_squared_error


class Model:

    def __init__(self, bias):
        self.bias = bias
        self.nb_rules = 0
        self.rules = []

    def get_bias(self):
        return self.bias

    def get_rules(self):
        return self.rules

    def add_rule(self, rule, g_update):
        self.nb_rules += 1
        self.rules.append((rule, g_update))

    def print_rules(self):
        print("################################## PRINTING RULES ##################################")
        for i, (rule, g_mean) in enumerate(self.rules):
            print("__________________________________RULE: {} ___________________________".format(i))
            for split in rule:
                print(split)

    def get_rule_length(self):
        rule_length_list = list(map(lambda e: len(e[0]), self.rules))
        rule_length_list.insert(0, int(1))
        return rule_length_list

    def count_fired_rules(self, example, index):
        count = 0
        for i in range(index+1):
            (rule, g_update) = self.rules[i]
            triggered_by_rule = True
            for index in range(len(rule)):
                split = rule[index]
                if example[split.feature] > split.value and split.type == "lt":
                    triggered_by_rule = False
                elif example[split.feature] <= split.value and split.type == "gt":
                    triggered_by_rule = False
            if triggered_by_rule:
                count += 1
        return count


    def get_prediction_for_example(self, example):
        prediction = self.bias
        for (rule, g_update) in self.rules:
            triggered_by_rule = True
            for index in range(len(rule)):
                split = rule[index]
                if example[split.feature] > split.value and split.type == "lt":
                    triggered_by_rule = False
                elif example[split.feature] <= split.value and split.type == "gt":
                    triggered_by_rule = False
            if triggered_by_rule:
                prediction += g_update
        return prediction

    def get_prediction_for_example_per_rule(self, example):
        prediction = self.bias
        prediction_list = np.array([prediction])
        for (rule, g_update) in self.rules:
            triggered_by_rule = True
            for index in range(len(rule)):
                split = rule[index]
                if example[split.feature] > split.value and split.type == "lt":
                    triggered_by_rule = False
                elif example[split.feature] <= split.value and split.type == "gt":
                    triggered_by_rule = False
            if triggered_by_rule:
                prediction += g_update
            prediction_list = np.append(prediction_list, prediction)
        return prediction_list

    def fit(self, test_data):
        """
        give final predictions of all the test_data examples (one prediciton after all the rules are executed)
        :param test_data:
        :return: a numpy list of predictions
        """
        predictions = []
        for example in test_data:
            predictions = predictions.append(self.get_prediction_for_example(example))
        return predictions

    def fit_per_rule(self, test_data):
        """
        give a list of predictions-lists of all the test_data examples (after each rule)
        :param test_data:
        :return: list of list predictions
        """
        predictions = np.array([np.repeat(0, self.nb_rules+1)])
        for example in test_data:
            next_pred = np.array([self.get_prediction_for_example_per_rule(example)])
            predictions = np.concatenate((predictions, next_pred))
        predictions = np.delete(predictions, 0, 0)
        return predictions

    def fit_loss_per_rule(self, test_data, test_target):
        loss_list = np.array([])
        prediction_matrix = self.fit_per_rule(test_data)
        for i in range(self.nb_rules+1):  # the +1 is for the 'bias rule'
            train_target = prediction_matrix[:,i]
            loss_list = np.append(loss_list, mean_squared_error(test_target, train_target))
        return loss_list

    def fit_loss_per_rule_and_triggered_rules(self, test_data, test_target):
        loss_list = np.array([])
        prediction_matrix = self.fit_per_rule(test_data)
        for i in range(self.nb_rules+1):  # the +1 is for the 'bias rule'
            train_target = prediction_matrix[:,i]
            loss_list = np.append(loss_list, mean_squared_error(test_target, train_target))
        best_model_index = np.argmin(loss_list) -1
        fired_list = []
        for example in test_data:
            fired_list.append(self.count_fired_rules(example, best_model_index))
        return loss_list, fired_list










