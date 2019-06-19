import sys
import os
from DecisionTree import DecisionTreeRegressor
from Data import Data



def main():
    print(sys.argv)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print("File {} does not exist.".format(filepath))
        exit(1)

    file = open(filepath)
    data = Data(file)
    #data.load_arff(file)
    #data.summary()

    #target_class = 'Class'
    #X, y = split_to_train_and_labels(data, target_class)
    #print(X, y)
    #print(data.data)

    # data = Data()
    # data.relation = 'data'
    # data.attributes = ['V1', 'V2', 'Class']
    # data.attribute_types = ['numeric', 'categorical', 'numeric']
    # data.attribute_values = ['numeric', ['1', '2', '3', '4', '5'], 'numeric']
    # data.n_attributes = 3
    # data.data = [[13, '1', 3], [22, '1', 5], [1, '2', 10], [1, '2', 20], [1, '3', 1], [1, '3', 1], [1, '5', 8], [1, '5', 6]]
    # data.summary()
    # tree = DecisionTreeRegressor()
    # tree.target_class = 'Class'
    # a = tree.create_categorical_split(data, 1)
    # print(a)



def split_to_train_and_labels(data, target_class):
    target_index = data.attributes.index(target_class)
    X = []
    y = []
    for instance in data.data:
        train = []
        for i in range(len(instance)):
            if i != target_index:
                train.append(instance[i])
            else:
                y.append(instance[i])
        X.append(train)
        #data.data = []
    return (X, y)

if __name__ == '__main__':
    main()