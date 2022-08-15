from src.regression import r_rule_based_gradient_boosting as v3_sampling_regression
import matplotlib.pyplot as plt
from sklearn import datasets, ensemble
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from time import process_time
from math import floor


plot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink"]

########################################################################################################################
#                                                   REGRESSION                                                         #
########################################################################################################################

def hyperparam_search():

    # loading the dataset
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 2500
    nb_bins = 10
    try:
        print("dataset: ", diabetes.data_filename)
    except:  # there are only two regression datasets, only on diabetes works 'data_filename'
        print("dataset: california housing")
    log = []
    log_loss = []
    for max_depth in [1,2,3,5,7]:
        for coarsity in [1,3,5,10]:
            for gain in [0, 0.1, 0.3, 0.5]:
                # rules
                print("start with params: max_depth = {}, coarsity = {}, gain_treshold = {}".format(max_depth, coarsity, gain))
                model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train,
                                                                            learning_rate=learning_rate,
                                                                            iterations=number_of_iterations,
                                                                            sample_size=sample_size,
                                                                            max_depth=max_depth,
                                                                            coarsity=coarsity,
                                                                            gain_treshold=gain,
                                                                            nb_buckets=nb_bins)
                loss = model.fit_loss_per_rule(X_test, y_test)
                log.append(str(np.min(loss)) + "achieved with params: max_depth = {}, coarsity = {}, gain_treshold = {}".format(max_depth, coarsity,gain))
                log_loss.append(str(np.min(loss)))
                print("minimal loss on test set =", str(np.min(loss)))
                print("___________________________________________________________________")

    # plotting
    for run in log:
        print(run)
    for loss in log_loss:
        print(loss)

def hyperparam_search_trees():

    # loading the dataset
    diabetes = datasets.fetch_california_housing()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 2500
    try:
        print("dataset: ", diabetes.filename)
    except:
        print("other dataset")
    log = []
    log_loss = []
    for max_depth in [1,2,3,4,5,7]:
        for coarsity in [2,3,5,10]:
            params = {
                "n_estimators": number_of_iterations,
                "max_depth": max_depth,
                "min_samples_leaf": coarsity,
                "learning_rate": learning_rate,
            }
            t2_start = process_time()
            reg = ensemble.GradientBoostingRegressor(**params, subsample=sample_size)
            reg.fit(X_train, y_train)
            number_of_rules = 0
            rules_per_tree = np.array([])
            for tree in reg.estimators_:
                number_of_rules += tree[0].get_n_leaves()
                if number_of_rules >= number_of_iterations:
                    break
                rules_per_tree = np.append(rules_per_tree, number_of_rules)
            test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
            for i, y_pred in enumerate(reg.staged_predict(X_test)):
                test_score[i] = reg.loss_(y_test, y_pred)

            log.append(str(np.min(test_score[:len(rules_per_tree)])) + "achieved with params: max_depth = {}, coarsity = {}, gain_treshold = {}".format(max_depth, coarsity, 0))
            log_loss.append(str(np.min(test_score[:len(rules_per_tree)])))
            print("minimal loss on test set =", str(np.min(test_score[:len(rules_per_tree)])))
            print("___________________________________________________________________")



    # plotting
    for run in log:
        print(run)
    for loss in log_loss:
        print(loss)

def binning_influence():
    # loading the dataset
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 2500
    max_depth= 5
    coarsity= 5
    log = []
    for bins in [10,25,50,100]:
        model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train,
                                                                    learning_rate=learning_rate,
                                                                    iterations=number_of_iterations,
                                                                    sample_size=sample_size,
                                                                    max_depth=max_depth,
                                                                    coarsity=coarsity,
                                                                    gain_treshold=0.5,
                                                                    nb_buckets=bins)
        print("with {} bins".format(bins))
        loss = model.fit_loss_per_rule(X_test, y_test)
        log.append(str(np.min(loss)) + " achieved with params: bins = {}, max_depth = {}, coarsity = {}".format(
            bins, max_depth, coarsity))
        print("minimal loss on test set =", str(np.min(loss)))
        print("___________________________________________________________________")
    # plotting
    for run in log:
        print(run)

def gain_influence():
    # loading the dataset
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 2500
    max_depth= 5
    coarsity= 1
    bins = 10
    log = []
    for gain_treshold in [0.3]:
        model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train,
                                                                    learning_rate=learning_rate,
                                                                    iterations=number_of_iterations,
                                                                    sample_size=sample_size,
                                                                    max_depth=max_depth,
                                                                    coarsity=coarsity,
                                                                    gain_treshold=gain_treshold,
                                                                    nb_buckets=bins)
        print("with {} gaintreshold".format(gain_treshold))
        loss = model.fit_loss_per_rule(X_test, y_test)
        log.append(str(np.min(loss)) + " achieved with params: gaintreshold = {}, max_depth = {}, coarsity = {}".format(
            gain_treshold, max_depth, coarsity))
        print("minimal loss on test set =", str(np.min(loss)))
        print("___________________________________________________________________")
    # plotting
    for run in log:
        print(run)


def regression_rules_vs_trees(with_train, with_nodes):

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 250
    max_depth = 5
    coarsity = 5
    nb_bins = 100
    gain = 0.5

    # loading the dataset
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0])*0.1), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    print("___________________________________________________________________")
    print("Parameters for rules: learning rate {}, sample size {}, depth {}, coarsity {}, nb_bins {}".format(learning_rate,
                                                                                                   sample_size,
                                                                                                   max_depth, coarsity,
                                                                                                   nb_bins))

    # rules
    t1_start = process_time()
    model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train,
                                                                learning_rate=learning_rate,
                                                                iterations=number_of_iterations,
                                                                sample_size=sample_size, max_depth=max_depth,
                                                                coarsity=coarsity,
                                                                nb_buckets=nb_bins,
                                                                gain_treshold=gain)
    loss = model.fit_loss_per_rule(X_test, y_test)
    if with_nodes:
        rule_length_list = np.array(model.get_rule_length())
        print(rule_length_list)
        nodes_list = np.cumsum(rule_length_list)
        plt.plot(nodes_list, loss, color=plot_colors[0], label="Regels: test")
        if with_train:
            plt.plot(nodes_list[1:], train_loss, '--', color=plot_colors[0], label="Regels: train")
    else:
        plt.plot(np.arange(len(loss)), loss, color=plot_colors[0], label="Regels: test")
        if with_train:
            plt.plot(np.arange(len(train_loss)),train_loss, '--',color=plot_colors[0], label="Regels: train")

    t1_stop = process_time()
    print("elapsed time for v3:", t1_stop - t1_start)
    print("Minimum loss with rules on the test set: ", np.min(loss))
    if with_nodes:
        print("Maximum reached at rule: ", nodes_list[np.argmin(loss)])
    else:
        print("Maximum reached at rule: ", np.argmin(loss))
    print("___________________________________________________________________")

    # trees
    params = {"n_estimators": number_of_iterations * 2, "max_depth": 2, "min_samples_leaf": 10,
              "learning_rate": learning_rate}

    t2_start = process_time()
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    number_of_rules = 0
    rules_per_tree = np.array([])
    for tree in reg.estimators_:
        number_of_rules += tree[0].get_n_leaves()
        if with_nodes:
            number_of_rules -= 1
            if number_of_rules >= np.sum(rule_length_list):
                break
        else:
            if number_of_rules >= number_of_iterations:
                break
        rules_per_tree = np.append(rules_per_tree, number_of_rules)

    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)
    plt.plot(rules_per_tree, test_score[:len(rules_per_tree)], color=plot_colors[1],label="Bomen: test")
    if with_train:
        plt.plot(rules_per_tree, reg.train_score_[:len(rules_per_tree)],'--', color=plot_colors[1], label="Bomen: train")
    t2_stop = process_time()
    print("elapsed time for v3:", t2_stop - t2_start)
    print("Minimum loss with trees on the test set: ", np.min(test_score[:len(rules_per_tree)]))
    print("Minimum reached at rule: ", rules_per_tree[np.argmin(test_score[:len(rules_per_tree)])])
    print("___________________________________________________________________")

    # plotting
    plt.title("Boosting vergelijking op california housing")
    plt.ylabel('fout')
    if with_nodes:
        plt.xlabel('#Splits toegevoegd')
    else:
        plt.xlabel('#Regels toegevoegd')
    plt.legend(loc="upper right")
    plt.show()


def regression_sample_size_scatter_plot(with_trees, with_nodes, number_of_runs):

    # set params the same for fair comparison
    learning_rate = 0.01
    number_of_iterations = 2500
    max_depth = 4
    coarsity = 5
    nb_bins = 10
    gain = 0.5

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    print("___________________________________________________________________")
    print("Parameters for rules: learning rate {}, depth {}, coarsity {}, nb_bins {}".format(
        learning_rate,
        max_depth, coarsity,
        nb_bins))
    print("___________________________________________________________________")

    for (iteration, sample) in enumerate([0.05, 0.1, 0.3, 0.5, 0.8, 1]):
        max_list_rules = []
        max_list_trees = []
        rule_list_rules = []
        rule_list_trees = []
        print("################ SAMPLE SIZE {} #################".format(sample))
        for i in range(number_of_runs):

            # rules
            model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train, learning_rate=learning_rate, sample_size=sample, iterations=number_of_iterations, max_depth=max_depth, nb_buckets=nb_bins, coarsity=coarsity, gain_treshold=gain)
            loss = model.fit_loss_per_rule(X_test, y_test)
            if with_nodes:
                rule_length_list = np.array(model.get_rule_length())
                nodes_list = np.cumsum(rule_length_list)
                max_list_rules.append(np.min(loss))
                rule_list_rules.append(nodes_list[np.argmin(loss)])
            else:

                max_list_rules.append(np.min(loss))
                rule_list_rules.append(np.argmin(loss))
            print("The min loss with rules and sample size {} is: {}".format(sample, np.min(loss)))

            # trees
            if with_trees:
                params = {
                    "n_estimators": number_of_iterations*2,
                    "max_depth": 3,
                    "min_samples_split": 10,
                    "learning_rate": learning_rate,
                }
                reg = ensemble.GradientBoostingRegressor(**params, subsample=sample)
                reg.fit(X_train, y_train)
                number_of_rules = 0
                rules_per_tree = np.array([])
                for tree in reg.estimators_:
                    number_of_rules += tree[0].get_n_leaves()
                    if with_nodes:
                        number_of_rules -= 1
                        if number_of_rules >= np.sum(rule_length_list):
                            break
                    else:
                        if number_of_rules >= number_of_iterations:
                            break

                    rules_per_tree = np.append(rules_per_tree, number_of_rules)
                test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
                for i, y_pred in enumerate(reg.staged_predict(X_test)):
                    test_score[i] = reg.loss_(y_test, y_pred)
                print("The min loss with trees and sample size {} is: {}".format(sample, np.min(test_score[:len(rules_per_tree)])))
                max_list_trees.append(np.min(test_score[:len(rules_per_tree)]))
                rule_list_trees.append(rules_per_tree[np.argmin(test_score[:len(rules_per_tree)])])
            if sample == 1:
                # sample size of 1 is deterministic
                break

        plt.scatter(rule_list_rules, max_list_rules, label="Regels: Sample grootte {}".format(sample), color=plot_colors[iteration])
        if with_trees:
            plt.scatter(rule_list_trees, max_list_trees, label="Bomen: Sample grootte {}".format(sample), marker='x', color=plot_colors[iteration])


    plt.grid(True)
    plt.legend(loc='best',prop={'size': 6})
    plt.ylabel("Minimale fout op test set")
    if with_nodes:
        plt.xlabel('#Splits gebruikt op minimale fout')
    else:
        plt.xlabel('#Regels gebruikt op minimale fout')
    plt.title("Vergelijking op diabetes met "+str(number_of_runs)+" uitvoeringen")
    plt.show()


def hyper_param_random_forest():
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    # X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    log = []
    for iterations in [50,100,150,200,250]:
        for max_depth in [1, 2, 3, 4, 5, 7]:
            for coarsity in [1,2, 3, 5, 10]:
                params = {
                    "n_estimators": iterations,
                    "max_depth": max_depth,
                    "min_samples_leaf": coarsity,
                }
                reg = ensemble.RandomForestRegressor(**params)
                reg.fit(X_train, y_train)
                number_of_rules = 0
                for tree in reg.estimators_:
                    number_of_rules += tree.get_n_leaves()
                loss = mean_squared_error(y_test, reg.predict(X_test))

                log.append(str(loss) + "achieved with params: iteration = {}, max_depth = {}, samples per leaf = {}".format(
                    iterations, max_depth, coarsity))
                print(loss)
                print("___________________________________________________________________")
            # plotting
    for run in log:
        print(run)

def regression_compare_other_algorithms(with_nodes, with_trees, number_of_runs):
    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 0.05
    number_of_iterations = 5000
    max_depth = 1
    coarsity = 10
    nb_bins = 10
    gain_treshold = 0

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

    print("___________________________________________________________________")
    print("Parameters for rules: learning rate {}, sample size {}, depth {}, coarsity {}, nb_bins {}".format(
        learning_rate,
        sample_size,
        max_depth, coarsity,
        nb_bins))
    print("___________________________________________________________________")


    min_list = []
    rule_list = []
    min_list_random = []
    rule_list_random = []
    min_list_trees = []
    rule_list_trees = []

    for i in range(number_of_runs):
        print("################ RUN {} #################".format(i))

        # rules
        model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train,
                                                                    learning_rate=learning_rate,
                                                                    sample_size=sample_size,
                                                                    iterations=number_of_iterations,
                                                                    coarsity=coarsity, nb_buckets=nb_bins,
                                                                    gain_treshold=gain_treshold,
                                                                    max_depth=max_depth)
        loss = model.fit_loss_per_rule(X_test, y_test)
        if with_nodes:
            rule_length_list = np.array(model.get_rule_length())
            nodes_list = np.cumsum(rule_length_list)
            min_list.append(np.min(loss))
            rule_list.append(nodes_list[np.argmin(loss)])
        else:

            min_list.append(np.min(loss))
            rule_list.append(np.argmin(loss))
        print("The min loss with rules is: {}".format(np.min(loss)))

        # trees
        if with_trees:
            params = {
                "n_estimators": number_of_iterations*2,
                "max_depth": 3,
                "min_samples_split": 10,
                "learning_rate": learning_rate,
                "subsample": 0.05,
            }
            reg = ensemble.GradientBoostingRegressor(**params)
            reg.fit(X_train, y_train)
            number_of_rules = 0
            rules_per_tree = np.array([])
            for tree in reg.estimators_:
                number_of_rules += tree[0].get_n_leaves()
                if number_of_rules >= number_of_iterations:
                    break
                rules_per_tree = np.append(rules_per_tree, number_of_rules)
            test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
            for i, y_pred in enumerate(reg.staged_predict(X_test)):
                test_score[i] = reg.loss_(y_test, y_pred)
            print("The min loss with trees is: {}".format( np.min(test_score)))
            min_list_trees.append(np.min(test_score[:len(rules_per_tree)]))
            rule_list_trees.append(rules_per_tree[np.argmin(test_score[:len(rules_per_tree)])])

        # random forest
        params = {
            "n_estimators": 150,
            "max_depth": 4,
        }
        reg = ensemble.RandomForestRegressor(**params)
        reg.fit(X_train, y_train)
        number_of_rules = 0
        for tree in reg.estimators_:
            number_of_rules += tree.get_n_leaves()
        loss = mean_squared_error(y_test, reg.predict(X_test))
        print("The min loss with random forest is: {}".format(loss))
        min_list_random.append(loss)
        rule_list_random.append(number_of_rules)


    plt.scatter(rule_list, min_list, label="src")
    plt.scatter(rule_list_trees, min_list_trees, label="XGBoost")
    plt.scatter(rule_list_random, min_list_random, label="Random forest")
    plt.title("Vergelijking van verschillende algoritmes op diabetes")
    plt.ylabel("Minimale fout op test set")
    plt.xlabel("#Regels gebruikt op minimale fout")
    plt.legend()
    plt.show()



def compare_optimal_models(number_of_runs):

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.1), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)


    max_list_rules = []
    max_list_trees = []
    rule_list_rules = []
    rule_list_trees = []
    node_list_rules = []
    node_list_trees = []

    for i in range(number_of_runs):
        print("___________________", i, "___________________" )
        # rules
        learning_rate = 0.01
        number_of_iterations = 2500
        max_depth = 1
        coarsity =10
        nb_bins = 10
        gain = 0
        sample = 0.05


        model, train_loss = v3_sampling_regression.RGBoostRegressor(X_train, y_train, learning_rate=learning_rate, sample_size=sample, iterations=number_of_iterations, max_depth=max_depth, nb_buckets=nb_bins, coarsity=coarsity, gain_treshold=gain)
        loss, fired_rules = model.fit_loss_per_rule_and_triggered_rules(X_test, y_test)
        rule_length_list = np.array(model.get_rule_length())
        nodes_list = np.cumsum(rule_length_list)
        # keep track of the bes values
        max_list_rules.append(np.min(loss))
        rule_list_rules.append(np.argmin(loss))
        node_list_rules.append(nodes_list[np.argmin(loss)])
        print("loss: {}, rules: {}, splits: {}, average fired rules: {}".format(np.min(loss), np.argmin(loss), nodes_list[np.argmin(loss)], np.sum(fired_rules)/y_test.shape[0]))

        # trees
        params = {
            "n_estimators": number_of_iterations*2,
            "max_depth": 3,
            "min_samples_split": 10,
            "learning_rate": learning_rate,
            "subsample":0.05
        }
        reg = ensemble.GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)
        number_of_rules = 0
        number_of_nodes = 0
        rules_per_tree = np.array([])
        nodes_per_tree = np.array([])
        for tree in reg.estimators_:
            number_of_rules += tree[0].get_n_leaves()
            number_of_nodes += tree[0].get_n_leaves() -1
            if number_of_nodes >= np.sum(rule_length_list):
                break
            rules_per_tree = np.append(rules_per_tree, number_of_rules)
            nodes_per_tree = np.append(nodes_per_tree, number_of_nodes)
        test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(reg.staged_predict(X_test)):
            test_score[i] = reg.loss_(y_test, y_pred)
        max_list_trees.append(np.min(test_score[:len(rules_per_tree)]))
        rule_list_trees.append(rules_per_tree[np.argmin(test_score[:len(rules_per_tree)])])
        node_list_trees.append(nodes_per_tree[np.argmin(test_score[:len(rules_per_tree)])])
        print("loss: {}, rules: {}, splits: {}, average fired rules: {}".format(np.min(test_score[:len(rules_per_tree)]), rules_per_tree[np.argmin(test_score[:len(rules_per_tree)])], nodes_per_tree[np.argmin(test_score[:len(rules_per_tree)])], np.argmin(test_score[:len(rules_per_tree)])))







regression_rules_vs_trees(True, False)