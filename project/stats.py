import os
import arff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')

smaller_than_500 = []
greater_than_500 = []
filenames = []

def load_results(filename):
    file_to_load = "results/" + filename[5:-5] + ".txt"
    file = open(file_to_load)
    result = {}
    result['name'] = filename[5:-5]
    file.readline()
    result['learning_rate'] = (float)(file.readline().split(':')[1].strip())
    result['max_leaf_nodes'] = (int)(file.readline().split(':')[1].strip())
    result['n_estimators'] = (int)(file.readline().split(':')[1].strip())
    result['subsample'] = (float)(file.readline().split(':')[1].strip())
    file.readline()
    result['accuracy'] = (float)(file.readline().split(':')[1].strip())
    return result

def plot_accuracies(accuracies):
    accuracies = sorted(accuracies, key=lambda accuracy: accuracy[1])
    labels = ["{:.2%}".format(acc[1]) for acc in accuracies]
    ax = pd.Series.from_array([accuracy[1] for accuracy in accuracies])
    ax = ax.plot(kind='barh', color="blue")
    ax.set_yticklabels([name[0] for name in accuracies])

    for i, v in enumerate(labels):
        ax.text(accuracies[i][1] + 0.01, i, v, va='center')

    plt.show()

def plot_all(learning_rate_counts, leaf_nodes_counts, n_estimators_counts, subsamping_counts):
    plt.subplot(1,4,1)
    objects = [learning_rate for learning_rate, count in learning_rate_counts.items()]
    y_pos = np.arange(len(objects))
    counts = [count for learning_rate, count in learning_rate_counts.items()]
    plt.bar(y_pos, counts, color="blue")
    plt.xticks(y_pos, objects)
    plt.ylabel('Counts')
    plt.xlabel('learning rate')
    plt.title("Learning rates")

    plt.subplot(1,4,2)
    objects = [nodes for nodes, count in leaf_nodes_counts.items()]
    y_pos = np.arange(len(objects))
    counts = [count for nodes, count in leaf_nodes_counts.items()]
    plt.bar(y_pos, counts, color="blue")
    plt.xticks(y_pos, objects)
    plt.title("Leaf nodes")
    plt.xlabel('number of leaf nodes')

    plt.subplot(1,4,3)
    objects = [trees for trees, count in n_estimators_counts.items()]
    y_pos = np.arange(len(objects))
    counts = [count for trees, count in n_estimators_counts.items()]
    plt.bar(y_pos, counts, color="blue")
    plt.xticks(y_pos, objects)
    plt.title("Trees")
    plt.xlabel('number of trees')

    plt.subplot(1,4,4)
    objects = [sample for sample, count in subsamping_counts.items()]
    y_pos = np.arange(len(objects))
    counts = [count for sample, count in subsamping_counts.items()]
    plt.bar(y_pos, counts, color="blue")
    plt.xticks(y_pos, objects)
    plt.title("Subsampling rates")
    plt.xlabel('rate')

    plt.show()

def plot_learning_rate(number_of_trees):

    i = 1
    for key in number_of_trees.keys():
        plt.subplot(1, 4, i)
        data = number_of_trees[key]
        objects = [key for key, count in data.items()]
        y_pos = np.arange(len(objects))
        counts = [count for key, count in data.items()]
        plt.bar(y_pos, counts, color="blue")
        plt.xticks(y_pos, objects)
        plt.title("Learning rate = {}".format(key))
        plt.xlabel('number of trees')

        i+=1

    plt.show()

#1st fig.
accuracies = []

#2nd fig.
learning_rate_counts = {0.1: 0, 0.05: 0, 0.01: 0, 0.001: 0}
leaf_nodes_counts = {2: 0, 3: 0, 6: 0, 11: 0}
n_estimators_counts = {100: 0, 200: 0, 500: 0, 1000: 0}
subsamping_counts = {0.6: 0, 0.7: 0, 0.8: 0, 1.0 : 0}

#3rd, 4th fig.
lower_learning_rate_counts = {0.1: 0, 0.05: 0, 0.01: 0, 0.001: 0}
lower_leaf_nodes_counts = {2: 0, 3: 0, 6: 0, 11: 0}
lower_n_estimators_counts = {100: 0, 200: 0, 500: 0, 1000: 0}
lower_subsamping_counts = {0.6: 0, 0.7: 0, 0.8: 0, 1.0 : 0}

upper_learning_rate_counts = {0.1: 0, 0.05: 0, 0.01: 0, 0.001: 0}
upper_leaf_nodes_counts = {2: 0, 3: 0, 6: 0, 11: 0}
upper_n_estimators_counts = {100: 0, 200: 0, 500: 0, 1000: 0}
upper_subsamping_counts = {0.6: 0, 0.7: 0, 0.8: 0, 1.0 : 0}

#5th fig.
number_of_trees = {0.1 : {100: 0, 200: 0, 500: 0, 1000: 0}, 0.05: {100: 0, 200: 0, 500: 0, 1000: 0}, 0.01: {100: 0, 200: 0, 500: 0, 1000: 0}, 0.001: {100: 0, 200: 0, 500: 0, 1000: 0}}

for file in os.listdir("data"):
    filename = os.path.join("data", file)
    print(filename)
    file = open(filename)
    dataset = arff.load(file)
    print(len(dataset['data']))
    result = load_results(filename)
    print(result)
    if len(dataset['data']) < 500:
        smaller_than_500.append(result['name'])
        lower_learning_rate_counts[result['learning_rate']] += 1
        lower_leaf_nodes_counts[result['max_leaf_nodes']] += 1
        lower_n_estimators_counts[result['n_estimators']] += 1
        lower_subsamping_counts[result['subsample']] += 1
    else:
        greater_than_500.append(result['name'])
        upper_learning_rate_counts[result['learning_rate']] += 1
        upper_leaf_nodes_counts[result['max_leaf_nodes']] += 1
        upper_n_estimators_counts[result['n_estimators']] += 1
        upper_subsamping_counts[result['subsample']] += 1
    filenames.append(filename)

    accuracies.append((result['name'], result['accuracy']))
    learning_rate_counts[result['learning_rate']] += 1
    leaf_nodes_counts[result['max_leaf_nodes']] += 1
    n_estimators_counts[result['n_estimators']] += 1
    subsamping_counts[result['subsample']] += 1
    
    number_of_trees[result['learning_rate']][result['n_estimators']] += 1
    
plot_accuracies(accuracies)
plot_all(learning_rate_counts, leaf_nodes_counts, n_estimators_counts, subsamping_counts)
plot_all(lower_learning_rate_counts, lower_leaf_nodes_counts, lower_n_estimators_counts, lower_subsamping_counts)
plot_all(upper_learning_rate_counts, upper_leaf_nodes_counts, upper_n_estimators_counts, upper_subsamping_counts)
plot_learning_rate(number_of_trees)
