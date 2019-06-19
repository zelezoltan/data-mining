import unittest
from DecisionTree import DecisionTreeRegressor, Node
from Data import Data

class TestDecisionTreeMethods(unittest.TestCase):

    def setUp(self):
        data = Data()
        data.relation = 'data'
        data.attributes = ['V1', 'V2', 'V3', 'Class']
        data.attribute_types = ['numeric', 'categorical', 'numeric', 'categorical']
        data.attribute_values = ['numeric', ['1', '2', '3', '4', '5'], 'numeric', ['1', '2']]
        data.n_attributes = 4
        data.data = [[13, '1', 3, '1'], [22, '1', 5, '2'], [17, '2', 10, '2'], [1, '2', 20, '2'], [5, '3', 1, '1'], [13, '3', 1, '1'], [2, '5', 8, '1'], [3, '5', 6, '1']]
        self.data = data
        #data.summary()
    @unittest.skip 
    def test_create_categorical_split(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'V3'
        attribute_index = 1
        splits = tree.create_categorical_split(self.data, attribute_index)
        # '1' => 4
        # '2' => 15
        # '3' => 1
        # '4' => no instance
        # '5' => 7
        print(splits)
        expected = [['3'], ['3', '1'], ['3', '1', '5']]
        self.assertListEqual(splits, expected)
    @unittest.skip 
    def test_create_numerical_split(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'Class'
        attribute_index = 0
        splits = tree.create_numeric_split(self.data, attribute_index)
        # sorted: 1, 2, 3, 5, 13, 13, 17, 22
        expected = [1.5, 2.5, 4, 9, 13, 15, 19.5]
        print(splits)
        self.assertListEqual(splits, expected)
    @unittest.skip 
    def test_generate_candidate_splits_for_categorical(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'V1'
        attribute_index = 1
        candidate_splits = tree.generate_candidate_splits(attribute_index, self.data)
        # '1' => (13 + 22) / 2 = 17.5
        # '2' => 9
        # '3' => 9
        # '4' => no instances
        # '5' => 2.5
        question_type = 'in'
        expected = (question_type, [['5'], ['5', '2'], ['5', '2', '3']])
        print(candidate_splits)
        self.assertTupleEqual(candidate_splits, expected)
    @unittest.skip 
    def test_generate_candidate_splits_for_numeric(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'Class'
        attribute_index = 2
        candidate_splits = tree.generate_candidate_splits(attribute_index, self.data)
        # sorted: 1, 1, 3, 5, 6, 8, 10, 20
        question_type = '<='
        expected = (question_type, [1, 2, 4, 5.5, 7, 9, 15])
        print(candidate_splits)
        self.assertTupleEqual(candidate_splits, expected)
    @unittest.skip 
    def test_find_best_split(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'V1'
        node = Node(data = self.data)
        split = tree.find_best_split(node)
        print(split)
        self.assertTupleEqual(('in', 1, ['5', '2', '3']), split[0])
        self.assertAlmostEqual(32.166666666, split[1])
    @unittest.skip 
    def test_018(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'V1'
        
        file = open("data/bank-marketing.arff")
        data = Data(file)
        #data.summary()
        #node = Node(data = data)
        #split = tree.find_best_split(node)
        #print(split)
        file.close()
    @unittest.skip 
    def test_mean_squared_error(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'V1'
        #split = ('<=', [2, 3, 4, 5, 6, 10])
        mse = tree.mean_squared_error(self.data)
        print(mse)
        self.assertAlmostEqual(53.5, mse)
    @unittest.skip     
    def test_impurity(self):
        tree = DecisionTreeRegressor()
        tree.target_class = 'V1'
        split = ('<=', 2, 5.5)
        impurity = tree.impurity(split, self.data)
        print(impurity)
        expected = 39.4375
        self.assertAlmostEqual(impurity, expected)

    def test_find_next_partition(self):
        tree = DecisionTreeRegressor(max_leaf_nodes=3)
        tree.target_class = 'V1'
        #(('in', 1, ['5', '2', '3']), 32.16666666666667)
        file = open("data/bank-marketing.arff")
        data = Data(file)
        node = Node(data = data)
        node.is_leaf = True
        tree.root = node
        change, next_node = tree.find_best_node_to_split(node)
        print(change, next_node)
        tree.partition(next_node)
        print(tree.n_leaves)
        tree.root.left_child.data.summary()
        
        tree.root.right_child.data.summary()
        tree.root.right_child.left_child.data.summary()
        tree.root.right_child.right_child.data.summary()

if __name__ == '__main__':
    unittest.main()