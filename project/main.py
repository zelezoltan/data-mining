import sys
import os
import arff
import pandas as pd
from math import log
import numpy as np
from GradientBoostingClassifier import GradientBoostingClassifier
from sklearn import metrics as ms
from sklearn.model_selection import train_test_split

def create_features_from_categorical(df, attributes):
    categorical_attributes = []
    for attribute in attributes:
        if attribute[1] != 'NUMERIC':
            attribute_name = attribute[0]
            categorical_attributes.append(attribute_name)
            if attribute_name == 'Class':
                continue
            attribute_values = attribute[1]
            for value in attribute_values:
                new_attribute = str(attribute_name) + "_is_" + (value)
                df[new_attribute] = df[attribute_name].apply(lambda x: 1.0 if x == value else 0.0)
    features = [attribute for attribute in df.columns if attribute not in categorical_attributes]
    target = 'Class'
    return df[features], df[target]


def load_data(path):
    if not os.path.isfile(path):
        print("File {} does not exist.".format(path))
        exit(1)

    file = open(path)
    dataset = arff.load(file)
    df = pd.DataFrame(dataset['data'])
    columns = [attribute[0] for attribute in dataset['attributes']]
    df.columns = columns
    
    return create_features_from_categorical(df, dataset['attributes'])

if __name__ == '__main__':
    
    if len(sys.argv) < 6:
        print("error")
        exit(1)
    filepath = sys.argv[1]

    # load the data
    features, target = load_data(filepath)

    # get hyperparameters
    n_estimators=int(sys.argv[2])
    learning_rate=float(sys.argv[3])
    max_leaf_nodes=int(sys.argv[4])
    subsample=float(sys.argv[5])
    
    # split to train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    # fit the model
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, subsample=subsample)
    clf.fit(X_train.values, y_train.values)
    predictions = clf.predict(X_test.values)

    # output accuracy
    print(ms.accuracy_score(y_test, predictions))
    
    
