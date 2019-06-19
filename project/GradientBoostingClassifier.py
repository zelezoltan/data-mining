from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier for binary classification
    """

    def __init__(self, n_estimators=100, subsample=1.0, learning_rate=0.1, max_leaf_nodes=None, max_features=None):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.treshold = 0.5
        self.trees = []
        self.tree_leaf_outputs = []
        self.target_classes = None
        self.initial_prediction = None

    def fit(self, X, y, verbose=False):
        log_odds, initial_probability = self.initial_guess(y)
        n_samples = len(y)
       
        predictions = np.full((n_samples, ), log_odds)
        self.initial_prediction = log_odds
        predicted_probabilities = np.full((n_samples, ), initial_probability) 
        
        observed_probabilities = np.array([(1.0 if label == self.target_classes[0] else 0.0) for label in y])
        
        for i in range(self.n_estimators):
            if verbose:
                loss = - np.sum(observed_probabilities * np.log(predicted_probabilities) + (1 - observed_probabilities) * np.log(1 - predicted_probabilities))
                print("Building tree " + str(i+1) + ", Loss: " + str(loss))
            
            # calculate the residuals
            residuals = observed_probabilities - predicted_probabilities

            # fit a tree to the residuals
            tree = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, max_features=self.max_features)

            # subsampling, stochastic gradient boosting
            train_set = X
            target_set = residuals
            if self.subsample < 1.0:
                train_set, target_set = self.sub_sample(train_set, target_set, n_samples)
            
            tree.fit(train_set, target_set)
            leaf_indices = tree.apply(X)

            # for each leaf calculate the output value for that leaf 
            leaf_outputs = self.calculate_leaf_outputs(residuals, leaf_indices, predicted_probabilities)
            self.trees.append(tree)
            self.tree_leaf_outputs.append(leaf_outputs)

            # make new prediction for each sample
            predicted_probabilities, predictions = self.calculate_new_predicitions(predictions, leaf_indices, leaf_outputs)

    def predict(self, X):
        y = self.predict_base(X)
        return np.array([(self.target_classes[0] if self.convert_to_probability(log_odds) > self.treshold else self.target_classes[1]) for log_odds in y])

    def predict_proba(self, X):
        y = self.predict_base(X)
        return np.array([self.convert_to_probability(log_odds) for log_odds in y])

    def predict_base(self, X):
        y = np.full((len(X), ), self.initial_prediction)
        for i in range(self.n_estimators):
            leaf_indices = self.trees[i].apply(X)
            leaf_outputs = np.array([self.tree_leaf_outputs[i][index] for index in leaf_indices])
            y = y + self.learning_rate * leaf_outputs
        return y

    def calculate_new_predicitions(self, previous_predictions, leaf_indices, leaf_outputs):
        new_predictions = []
        new_probabilities = []
        for i in range(len(previous_predictions)):
            leaf_index = leaf_indices[i]
            leaf_value = leaf_outputs[leaf_index]
            new_prediction = previous_predictions[i] + self.learning_rate * leaf_value
            new_predictions.append(new_prediction)
            new_probabilities.append(self.convert_to_probability(new_prediction))
        
        return np.array(new_probabilities), np.array(new_predictions)

    def calculate_leaf_outputs(self, residuals, leaf_indices, predicted_probabilities):
        unique = np.unique(leaf_indices)
        leaf_nominators = {index: 0 for index in unique}
        leaf_denominators = {index: 0 for index in unique}
        for i in range(len(residuals)):
            leaf_index = leaf_indices[i]
            leaf_nominators[leaf_index] += residuals[i]
            leaf_denominators[leaf_index] += predicted_probabilities[i] * (1 - predicted_probabilities[i])
        
        return {index: leaf_nominators[index] / leaf_denominators[index] for index in unique}

    def sub_sample(self, train_set, target_set, n_samples):
        indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace = False)
        return train_set[indices], target_set[indices]

    def calculate_residuals(self, observed_probabilities, predicted_probabilities):
        return np.array([x[0] - x[1] for x in zip(observed_probabilities, predicted_probabilities)])

    def initial_guess(self, y):
        self.target_classes, counts = np.unique(y, return_counts=True)
        log_odds = np.log(counts[0] / counts[1])
        return log_odds, self.convert_to_probability(log_odds)

    def convert_to_probability(self, log_odds):
        return np.exp(log_odds) / (1 + np.exp(log_odds))