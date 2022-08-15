from src.classification import c_rule_based_gradient_boosting as v3_sampling_classification
import matplotlib.pyplot as plt
from sklearn import datasets, ensemble
import numpy as np
from sklearn.model_selection import train_test_split
from time import process_time
from math import floor
from mlrl.boosting import Boomer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

plot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink"]


########################################################################################################################
#                                           CLASSIFICATION                                                             #
########################################################################################################################


def make_binary_classification(y, class_label):
    for (i,y_label) in enumerate(y):
        if y_label == class_label:
            y[i] = 1
        else:
            y[i] = 0
    return y


def hyperparam_search():

    # loading the dataset
    print('iris')
    diabetes = datasets.load_iris()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    y = make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 250
    nb_bins = 10
    try:
        print("dataset: ", diabetes.filename)
    except:
        print("other dataset")
    log = []
    log_loss = []
    for max_depth in [1,2,3,5,7]:
        for coarsity in [1,3,5,10]:
            for gain in [0, 0.1, 0.3,0.5]:
                # rules
                print("start with params: max_depth = {}, coarsity = {}, gain_treshold = {}".format(max_depth, coarsity, gain))
                model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train,
                                                                                 learning_rate=learning_rate,
                                                                                 iterations=number_of_iterations,
                                                                                 sample_size=sample_size,
                                                                                 max_depth=max_depth,
                                                                                 coarsity=coarsity,
                                                                                 gain_treshold=gain,
                                                                                 nb_buckets=nb_bins)
                loss = model.get_accuracy_per_rule(X_test, y_test)
                log.append(str(np.max(loss)) + "achieved with params: max_depth = {}, coarsity = {}, gain_treshold = {}".format(max_depth, coarsity,gain))
                log_loss.append(str(np.max(loss)))
                print("minimal loss on test set =", str(np.max(loss)))
                print("___________________________________________________________________")

    # plotting
    for run in log:
        print(run)
    for loss in log_loss:
        print(loss)

def hyperparam_search_trees():

    # loading the dataset
    diabetes = datasets.fetch_covtype()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    X, y = diabetes.data[idx], diabetes.target[idx]
    y = make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

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
            reg = ensemble.GradientBoostingClassifier(**params, subsample=sample_size)
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
                test_score[i] = get_accuracy(y_test, y_pred)

            log.append(str(np.max(test_score[:len(rules_per_tree)])) + "achieved with params: max_depth = {}, coarsity = {}, gain_treshold = {}".format(max_depth, coarsity, 0))
            log_loss.append(str(np.max(test_score[:len(rules_per_tree)])))
            print("minimal loss on test set =", str(np.max(test_score[:len(rules_per_tree)])))
            print("___________________________________________________________________")



    # plotting
    for run in log:
        print(run)
    for loss in log_loss:
        print(loss)



def binning_influence():
    # loading the dataset
    diabetes = datasets.fetch_covtype()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    X, y = diabetes.data[idx], diabetes.target[idx]
    y = make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 2500
    max_depth= 3
    coarsity= 10
    log = []
    for bins in [10,25,50,100]:
        model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train,
                                                                         learning_rate=learning_rate,
                                                                         iterations=number_of_iterations,
                                                                         sample_size=sample_size,
                                                                         max_depth=max_depth,
                                                                         coarsity=coarsity,
                                                                         gain_treshold=0.5,
                                                                         nb_buckets=bins)
        print("with {} bins".format(bins))
        loss = model.get_accuracy_per_rule(X_test, y_test)
        log.append(str(np.max(loss)) + " achieved with params: bins = {}, max_depth = {}, coarsity = {}".format(
            bins, max_depth, coarsity))
        print("minimal loss on test set =", str(np.max(loss)))
        print("___________________________________________________________________")
    # plotting
    for run in log:
        print(run)


def gain_influence():
    # loading the dataset
    print("iris")
    diabetes = datasets.fetch_covtype()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001),
                           replace=False)
    X, y = diabetes.data[idx], diabetes.target[idx]
    y = make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 1
    number_of_iterations = 2500
    max_depth = 1
    coarsity = 10
    bins = 10
    log = []
    for gain_treshold in [0,0.0001,0.001,0.01]:
        model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train,
                                                                         learning_rate=learning_rate,
                                                                         iterations=number_of_iterations,
                                                                         sample_size=sample_size,
                                                                         max_depth=max_depth,
                                                                         coarsity=coarsity,
                                                                         gain_treshold=gain_treshold,
                                                                         nb_buckets=bins)
        print("with {} gaintreshold".format(gain_treshold))
        loss = model.get_accuracy_per_rule(X_test, y_test)
        log.append(str(np.max(loss)) + " achieved with params: gaintreshold = {}, max_depth = {}, coarsity = {}, bins = {}".format(
            gain_treshold, max_depth, coarsity, bins))
        print("maximal accuracy on test set =", str(np.max(loss)))
        print("___________________________________________________________________")
    # plotting
    for run in log:
        print(run)



def classification_rules_vs_trees(with_train, with_nodes):

    # set params the same for fair comparison
    learning_rate = 0.01
    sample_size = 0.8
    number_of_iterations = 500
    max_depth = 3
    coarsity = 10
    nb_bins = 10
    gain = 0

    print("Parameters: learning rate {}, sample size {}, depth {}, coarsity {}, nb_bins {}".format(learning_rate, sample_size, max_depth, coarsity, nb_bins))

    diabetes = datasets.load_digits()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001),
                           replace=False)
    # X, y = diabetes.data[idx], diabetes.target[idx]
    make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    # rules
    t1_start = process_time()
    model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train,
                                                                     learning_rate=learning_rate,
                                                                     sample_size=sample_size,
                                                                     iterations=number_of_iterations,
                                                                     max_depth = max_depth,
                                                                     nb_buckets = nb_bins,
                                                                     coarsity=coarsity,
                                                                     gain_treshold=gain)
    loss = model.get_accuracy_per_rule(X_test, y_test)
    if with_nodes:
        rule_length_list = model.get_rule_length()
        print(rule_length_list)
        nodes_list = np.cumsum(rule_length_list)
        plt.plot(nodes_list, loss, color=plot_colors[0],label="Regels: test")
        if with_train:
            plt.plot(nodes_list[1:], train_loss, '--', color=plot_colors[0],label="Regels: train")
    else:
        plt.plot(np.arange(len(loss)), loss, color=plot_colors[0],label="Regels: test")
        if with_train:
            plt.plot(np.arange(len(train_loss)), train_loss, '--', color=plot_colors[0],label="Regels: train")
    t1_stop = process_time()
    print("elapsed time for v3:", t1_stop - t1_start)
    print("Maximum accuracy on test set with rules: ", np.max(loss))
    if with_nodes:
        print("Maximum reached at rule: ", nodes_list[np.argmax(loss)])
    else:
        print("Maximum reached at rule: ", np.argmax(loss))
    print("___________________________________________________________________")

    # trees
    params = {
        "n_estimators": number_of_iterations*4,
        "max_depth": 1,
        "min_samples_split": 10,
        "learning_rate": learning_rate,
    }
    t2_start = process_time()
    reg = ensemble.GradientBoostingClassifier(**params, subsample=0.1)
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
        test_score[i] = get_accuracy(y_test, y_pred)
    plt.plot(rules_per_tree, test_score[:len(rules_per_tree)], color=plot_colors[1],label="Bomen: test")


    if with_train:
        train_score = np.zeros((params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(reg.staged_predict(X_train)):
            train_score[i] = get_accuracy(y_train, y_pred)
        plt.plot(rules_per_tree, train_score[:len(rules_per_tree)], '--', color=plot_colors[1],label="Bomen: train")

    t2_stop = process_time()
    print("elapsed time for v3:", t2_stop - t2_start)
    print("Maximum accuracy on test set with trees: ", np.max(test_score[:len(rules_per_tree)]))
    print("Maximum reached at rule: ",  rules_per_tree[np.argmax(test_score[:len(rules_per_tree)])])
    print("___________________________________________________________________")

    # plotting
    plt.title("Boosting vergelijking op iris (versicolour)")
    plt.ylabel('Accuraatheid')
    if with_nodes:
        plt.xlabel('#Splits toegevoegd')
    else:
        plt.xlabel('#regels toegevoegd')
    plt.legend(loc="best")
    plt.show()


def check_best_sample_graph_classification():
    diabetes = datasets.load_breast_cancer()
    X, y = diabetes.data, diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=4
    )
    samples = [0.05, 0.1, 0.3, 0.5, 0.8, 1]

    for sample_size in samples:
        model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train, 0.01, sample_size=sample_size)
        loss = model.get_accuracy_per_rule(X_test, y_test)
        plt.plot(np.arange(len(loss)), loss, label="Sample size {} ".format(sample_size))
        print("The max accuracy with sample size {} is: {}".format(sample_size, max(loss)))


    plt.title("Sample size comparison")
    plt.ylabel('Accuracy')
    plt.xlabel('Rules added')
    plt.legend(loc="best")
    plt.show()


def classification_sample_size_scatter_plot(with_trees, with_nodes, number_of_runs):

    # set params the same for fair comparison
    learning_rate = 0.01
    number_of_iterations = 500
    max_depth = 3
    coarsity = 10
    nb_bins = 10
    gain = 0

    print("Parameters: learning rate {}, depth {}, coarsity {}, nb_bins {}".format(learning_rate,

                                                                                                   max_depth, coarsity,
                                                                                                   nb_bins))
    diabetes = datasets.load_digits()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    make_binary_classification(y,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    print(y_test.shape)
    for (iteration, sample) in enumerate([0.05, 0.1, 0.3, 0.5, 0.8, 1]):
        max_list_rules = []
        max_list_trees = []
        rule_list_rules = []
        rule_list_trees = []
        print("################ SAMPLE SIZE {} #################".format(sample))
        for i in range(number_of_runs):

            # rules
            model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train, learning_rate=learning_rate, sample_size=sample, iterations=number_of_iterations, max_depth=max_depth, nb_buckets=nb_bins, coarsity=coarsity, gain_treshold=gain)
            loss = model.get_accuracy_per_rule(X_test, y_test)
            if with_nodes:
                rule_length_list = np.array(model.get_rule_length())
                nodes_list = np.cumsum(rule_length_list)
                max_list_rules.append(np.max(loss))
                rule_list_rules.append(nodes_list[np.argmax(loss[:len(nodes_list)])])
            else:

                max_list_rules.append(np.max(loss))
                rule_list_rules.append(np.argmax(loss))

            print("The max accuracy  with rules and sample size {} is: {}".format(sample, np.max(loss)))

            # trees
            if with_trees:
                params = {
                    "n_estimators": number_of_iterations*2,
                    "max_depth": 1,
                    "min_samples_split": 10,
                    "learning_rate": learning_rate,
                }
                reg = ensemble.GradientBoostingClassifier(**params, subsample=sample)
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
                    test_score[i] = get_accuracy(y_test, y_pred)
                print("The max accuracy with trees and sample size {} is: {}".format(sample, np.max(test_score[:len(rules_per_tree)])))
                max_list_trees.append(np.max(test_score[:len(rules_per_tree)]))
                rule_list_trees.append(rules_per_tree[np.argmax(test_score[:len(rules_per_tree)])])

            if sample == 1:
                # sample size of 1 is deterministic
                break

        plt.scatter(rule_list_rules, max_list_rules, label="Regels:  sample grootte {}".format(sample), color=plot_colors[iteration])
        if with_trees:
            plt.scatter(rule_list_trees, max_list_trees, label="Bomen: sample grootte {}".format(sample), marker='x', color=plot_colors[iteration])

    plt.legend(loc='best', prop={'size': 6})
    plt.grid(True)
    plt.ylabel("Maximale accuraatheid op test set")
    if with_nodes:
        plt.xlabel("#splits gebruikt op maximale accuraatheid")
    else:
        plt.xlabel("#Regels gebruikt op maximale accuraatheid")
    plt.title("Vergelijking op digits (1) met "+str(number_of_runs)+" uitvoeringen")
    plt.show()


def hyper_param_random_forest():
    diabetes = datasets.load_iris()
    X, y = diabetes.data, diabetes.target
    make_binary_classification(y, 1)
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    log = []
    for iterations in [100,150,200,250,500]:
        for max_depth in [1, 2, 3, 4, 5, 7]:
            for coarsity in [1,2, 3, 5, 10]:
                params = {
                    "n_estimators": iterations,
                    "max_depth": max_depth,
                    "min_samples_leaf": coarsity,
                }
                reg = ensemble.RandomForestClassifier(**params)

                reg.fit(X_train, y_train)
                number_of_rules = 0
                for tree in reg.estimators_:
                    number_of_rules += tree.get_n_leaves()
                loss = get_accuracy(y_test, reg.predict(X_test))

                log.append(str(loss) + "achieved with params: iteration = {}, max_depth = {}, samples per leaf = {}".format(
                    iterations, max_depth, coarsity))
                print(loss)
                print("___________________________________________________________________")
            # plotting
    for run in log:
        print(run)

def hyper_param_boomer():
    diabetes = datasets.load_iris()
    X, y = diabetes.data, diabetes.target
    make_binary_classification(y, 1)
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    log = []
    for max_depth in [1, 2, 3, 4, 5, 7]:
        for coarsity in [1,2, 3, 5, 10]:

            reg = Boomer(max_rules=250, head_type='single-label', shrinkage=0.01, rule_induction="top-down{max_conditions="+str(max_depth)+",min_coverage="+str(coarsity)+"}")
            y_train_zeros = np.zeros(y_train.shape[0])
            reg.fit(X_train, list(zip(y_train, y_train_zeros)))
            y_pred = reg.predict(X_test)
            acc_boomer = get_accuracy(y_test, list(zip(*y_pred))[0])

            log.append(str(acc_boomer) + "achieved with params: max_depth = {}, samples per leaf = {}".format( max_depth, coarsity))
            print(acc_boomer)
            print("___________________________________________________________________")
            # plotting
    for run in log:
        print(run)

def compare_boomer():
    diabetes = datasets.load_iris()
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    #X, y = diabetes.data[idx], diabetes.target[idx]
    make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


    # Rules
    learning_rate = 0.01
    sample_size = 0.8
    number_of_iterations = 250
    max_depth = 2
    coarsity = 10
    nb_bins = 10
    gain = 0
    model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train,
                                                                     learning_rate=learning_rate,
                                                                     sample_size=sample_size,
                                                                     iterations=number_of_iterations,
                                                                     max_depth = max_depth,
                                                                     nb_buckets = nb_bins,
                                                                     coarsity=coarsity,
                                                                     gain_treshold=gain)
    loss = model.get_accuracy_per_rule(X_test, y_test)
    acc_rules = np.max(loss)

    # Boomer
    reg = Boomer(max_rules=number_of_iterations, head_type='single-label', shrinkage=0.01)
    y_train_zeros = np.zeros(y_train.shape[0])
    reg.fit(X_train, list(zip(y_train, y_train_zeros)))
    y_pred = reg.predict(X_test)
    acc_boomer = get_accuracy(y_test, list(zip(*y_pred))[0])

    # Trees
    params = {
        "n_estimators": number_of_iterations * 4,
        "max_depth": 1,
        "min_samples_leaf": 10,
        "learning_rate": learning_rate,
        "subsample": 0.8,
    }
    reg = ensemble.GradientBoostingClassifier(**params)
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
        test_score[i] = get_accuracy(y_test, y_pred)
    acc_trees = np.max(test_score[:len(rules_per_tree)])

    #Random Forest
    params = {
        "n_estimators": 100,
        "max_depth": 2,
        "min_samples_leaf": 1,
    }
    reg = ensemble.RandomForestClassifier(**params)
    reg.fit(X_train, y_train)
    number_of_rules = 0
    number_of_splits = 0
    for tree in reg.estimators_:
        number_of_rules += tree.get_n_leaves()
        number_of_splits += tree.get_n_leaves()-1
    acc_random = get_accuracy(y_test, reg.predict(X_test))

    print("Rules:", acc_rules)
    print("Trees:", acc_trees)
    print("Boomer:", acc_boomer)
    print("Random:", acc_random, "number_of_rules", number_of_rules, "number_of_splits", number_of_splits)

def make_bar_chart():
    labels = ['Iris', 'Digits', 'Breast Cancer', 'Covtype']
    rules = [1, 0.983, 0.982, 0.803]
    trees = [1, 0.944, 0.964, 0.812]
    boomer = [0.933,0.931,0.982, 0.726]
    random = [1, 0.983, 0.912, 0.70]

    N = 4
    x = np.arange(N)
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, rules, width, label='src')
    rects2 = ax.bar(x + width, trees, width, label='XGBoost')
    rects3 = ax.bar(x + width*2, boomer, width, label='Boomer' )
    rects4 = ax.bar(x + width*3, random, width, label='Random Forest')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Minimale accuraatheid')
    ax.set_title('Vergelijking src met andere algoritmes')
    ax.set_xticks(x+1.5*width, labels)
    ax.legend()
    ax.set_ylim([0.6, 1.05])

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    plt.show()



def compare_optimal_models(number_of_runs):
    diabetes = datasets.fetch_covtype()
    print("covtype")
    X, y = diabetes.data, diabetes.target
    np.random.seed(0)
    idx = np.random.choice(np.arange(diabetes.target.shape[0]), floor((diabetes.target.shape[0]) * 0.001), replace=False)
    X, y = diabetes.data[idx], diabetes.target[idx]
    make_binary_classification(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

    max_list_rules = []
    max_list_trees = []
    rule_list_rules = []
    rule_list_trees = []
    node_list_rules = []
    node_list_trees = []

    for i in range(number_of_runs):
        print("___________________", i, "___________________")
        # rules
        learning_rate = 0.01
        number_of_iterations = 2500
        max_depth = 3
        coarsity = 10
        nb_bins = 10
        gain = 0.5
        sample = 0.5

        model, train_loss = v3_sampling_classification.RGBoostClassifier(X_train, y_train,
                                                                         learning_rate=learning_rate,
                                                                         sample_size=sample,
                                                                         iterations=number_of_iterations,
                                                                         max_depth=max_depth,
                                                                         nb_buckets=nb_bins,
                                                                         coarsity=coarsity,
                                                                         gain_treshold=gain)
        loss, fired_rules = model.get_accuracy_and_triggered_rules(X_test, y_test)
        rule_length_list = np.array(model.get_rule_length())
        nodes_list = np.cumsum(rule_length_list)
        # keep track of the bes values
        max_list_rules.append(np.max(loss))
        rule_list_rules.append(np.argmax(loss))
        node_list_rules.append(nodes_list[np.argmax(loss)])
        print("loss: {}, rules: {}, splits: {}, average fired rules: {}".format(np.max(loss), np.argmax(loss),
                                                                                nodes_list[np.argmax(loss)],
                                                                                np.sum(fired_rules) / y_test.shape[
                                                                                    0]))

        # trees
        params = {
            "n_estimators": number_of_iterations * 4,
            "max_depth": 2,
            "min_samples_split": 5,
            "learning_rate": learning_rate,
            "subsample": 0.05
        }
        reg = ensemble.GradientBoostingClassifier(**params)
        reg.fit(X_train, y_train)
        number_of_rules = 0
        number_of_nodes = 0
        rules_per_tree = np.array([])
        nodes_per_tree = np.array([])
        for tree in reg.estimators_:
            number_of_rules += tree[0].get_n_leaves()
            number_of_nodes += tree[0].get_n_leaves() - 1
            if number_of_nodes >= np.sum(rule_length_list):
                break
            rules_per_tree = np.append(rules_per_tree, number_of_rules)
            nodes_per_tree = np.append(nodes_per_tree, number_of_nodes)
        test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(reg.staged_predict(X_test)):
            test_score[i] = get_accuracy(y_test, y_pred)
        max_list_trees.append(np.max(test_score[:len(rules_per_tree)]))
        rule_list_trees.append(rules_per_tree[np.argmax(test_score[:len(rules_per_tree)])])
        node_list_trees.append(nodes_per_tree[np.argmax(test_score[:len(rules_per_tree)])])
        print("loss: {}, rules: {}, splits: {}, average fired rules: {}".format(
            np.max(test_score[:len(rules_per_tree)]), rules_per_tree[np.argmax(test_score[:len(rules_per_tree)])],
            nodes_per_tree[np.argmax(test_score[:len(rules_per_tree)])],
            np.argmax(test_score[:len(rules_per_tree)])))


def get_accuracy(y_test, y_predict):
    return 1 - (np.sum(np.absolute(np.subtract(y_test, y_predict))) / y_test.shape[0])

classification_rules_vs_trees(True, False)