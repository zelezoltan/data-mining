import unittest
import numpy as np
from GradientBoostingClassifier import GradientBoostingClassifier

class TestDecisionTreeMethods(unittest.TestCase):

    def test_initial_guess(self):
        print('---------- test_initial_guess ---------')
        print()
        y = np.array(['1', '1', '2', '2', '1', '1'])
        expected_log_odds = 0.69314718056
        expected_probability = 0.66666666
        clr = GradientBoostingClassifier()
        log_odds, probability = clr.initial_guess(y)
        print("expected log(odds): " + str(expected_log_odds) + ", actual: " + str(log_odds))
        print("expected probability: " + str(expected_probability) + ", actual: " + str(probability))
        self.assertAlmostEqual(expected_log_odds, log_odds)
        self.assertAlmostEqual(expected_probability, probability)
        print()

    @unittest.skip
    def test_calculate_residuals(self):
        print('---------- test_calculate_residuals ---------')
        print()
        observed = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        y = np.array(['1', '1', '2', '2', '1', '1'])
        clr = GradientBoostingClassifier()
        
        predicted_probabilities = np.array([0.2, 0.3, 0.8, 0.2, 0.1, 0.7])
        expected_residuals = np.array([0.8, 0.7, -0.8, -0.2, 0.9, 0.3])
        residuals = clr.calculate_residuals(observed, predicted_probabilities)
        print(residuals, expected_residuals)
        print()
        
    def test_sub_sample(self):
        print('---------- test_sub_sample ---------')
        print()
        clr = GradientBoostingClassifier(subsample=0.5)
        train_set = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        target_set = np.array(['1', '2', '1', '2'])
        train, target = clr.sub_sample(train_set, target_set, 4)
        print("train set: " + str(train_set))
        print("target set: " + str(target_set))
        print("sampled train set: " + str(train))
        print("sample target set:" + str(target))
        self.assertEqual(2, len(train))
        self.assertEqual(2, len(target))
        print()

    def test_calculate_leaf_outputs(self):
        print('---------- test_calculate_leaf_outputs ---------')
        print()
        residuals = np.array([0.1, 0.5, -0.5, -0.1, 0.1, 0.1])
        leaf_indices = np.array([11, 7, 2, 11, 11, 11])
        predicted_probabilities = np.array([0.9, 0.5, 0.5, 0.1, 0.9, 0.9])
        clr = GradientBoostingClassifier()
        leaf_outputs = clr.calculate_leaf_outputs(residuals, leaf_indices, predicted_probabilities)
        expected_leaf_outputs = {2: -2.0, 7: 2.0, 11: 0.555555556}
        print("expected leaf outputs: " + str(expected_leaf_outputs) + ", actual: " + str(leaf_outputs))
        #self.assertDictEqual(expected_leaf_outputs, leaf_outputs)
        print()

    def test_calculate_new_predictions(self):
        print('---------- test_calculate_new_predictions ---------')
        print()
        previous_predictions = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        leaf_indices = [3, 2, 2, 1, 3, 3]
        leaf_outputs = {1: -3.3, 2: -1.0, 3: 1.4}
        expected_predictions = [1.82, -0.1, -0.1, -1.94, 1.82, 1.82]
        expected_probabilities = [0.9, 0.5, 0.5, 0.1, 0.9, 0.9]
        clr = GradientBoostingClassifier(learning_rate=0.8)
        probabilities, predictions = clr.calculate_new_predicitions(previous_predictions, leaf_indices, leaf_outputs)
        print("expected predictions: " + str(expected_predictions) + ", actual: " + str(predictions))
        print("expected probabilities: " + str(expected_probabilities) + ", actual: " + str(probabilities))
        print()

    def test_predict(self):
        print('---------- test_predict ---------')
        print()
        class TreeStub1:
            def apply(self, X):
                return np.array([3])

        class TreeStub2:
            def apply(self, X):
                return np.array([2])

        clr = GradientBoostingClassifier(n_estimators=2, learning_rate=0.8)
        clr.initial_prediction = 0.7
        clr.target_classes = np.array(['yes', 'no'])
        clr.treshold = 0.5
        tree_leaf_output_1 = {1: -3.3, 2: -1.0, 3: 1.4}
        tree_leaf_output_2 = {1: -2.0, 2: 0.6, 3: 2.0}
        tree = TreeStub1()
        clr.trees.append(tree)
        tree = TreeStub2()
        clr.trees.append(tree)
        clr.tree_leaf_outputs.append(tree_leaf_output_1)
        clr.tree_leaf_outputs.append(tree_leaf_output_2)

        X = [[1, 1, 1]]
        y = clr.predict(X)
        proba = clr.predict_proba(X)
        print("expected outputs: ['yes'], actual: " + str(y))
        print("expected probabilities: [0.908877039], actual: " + str(proba))
        self.assertListEqual(y.tolist(), ['yes'])
        self.assertAlmostEqual(proba[0], 0.908877039)
        print()

if __name__ == '__main__':
    unittest.main()