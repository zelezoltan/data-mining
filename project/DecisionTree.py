from Data import Data
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy
from itertools import repeat

class DecisionTreeRegressor(object):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.root = None
        self.data = None
        self.target_class = None
        self.min_samples_leaf = min_samples_leaf
        self.n_leaves = 1

    def fit(self, data, target_class):
        self.data = data
        self.target_class = target_class
        self.root = Node(data)
        self.root.is_leaf = True
        #self.partition(self.root)
        self.find_next_partition()

    def find_next_partition(self):
        node = self.root
        change, node = self.find_best_node_to_split(node)
        if change > 0:
            return
        self.partition(node)
    
    def find_best_node_to_split(self, node):
        if node.is_leaf:
            node_impurity = self.mean_squared_error(node.data)
            split, split_impurity = self.find_best_split(node)
            #split_impurity = self.impurity(split, node.data)
            impurity_change = split_impurity - node_impurity
            return impurity_change, node
        else:
            best_change_left, left_node = self.find_best_node_to_split(node.left_child)
            best_change_right, right_node = self.find_best_node_to_split(node.right_child)
            if best_change_left < best_change_right:
                return best_change_left, left_node
            else:
                return best_change_right, right_node
        

    def partition(self, node):
        if self.max_leaf_nodes is not None and self.n_leaves + 1  > self.max_leaf_nodes:
            return
        
        if len(node.data.data) < self.min_samples_split:
            return
        print("splitted")
        split, mse = self.find_best_split(node)
        left_node, right_node = self.split_data(split, node.data)
        node.is_leaf = False
        left_node.is_leaf = True
        right_node.is_leaf = True
        self.n_leaves += 1
        node.split = split
        left_node.parent = node
        right_node.parent = node
        node.left_child = left_node
        node.right_child = right_node

        self.find_next_partition()

    def find_best_split(self, node):
        """
        Finds the best split among all possible splits.
        To determine the "goodness" of a split, it uses the impurity function.
        The lower the impurity the better.

        Returns a tuple with the best found split and the corresponding mean
        squared error. E.g.: (('in', 1, ['1', '3']), 30.45)
        The first element of the split is the question type ('<=' or 'in'),
        the socond is the index of the attribute to split and the third is 
        the found values to split on.
        """
        best_split = None

        possible_splits = []
        for index, attribute in enumerate(node.data.attributes):
            if attribute != self.target_class:
                candidate_splits = self.generate_candidate_splits(index, node.data)
                for split in candidate_splits[1]:
                    possible_splits.append((candidate_splits[0], index, split))
        print(possible_splits)
        # using 4 threads to calculate the impurities for all the splits
        #pool = ThreadPool(4)
        #impurities = pool.starmap(self.impurity, zip(possible_splits, repeat(node.data, len(possible_splits))))
        #pool.close()
        #pool.join()
        impurities = []
        for split in possible_splits:
            impurities.append(self.impurity(split, node.data))
        print(impurities)
        index = None
        best_mse = None
        for i, mse in enumerate(impurities):
            if index is None:
                index = i
                best_mse = mse
            if mse < best_mse:
                index = i
                best_mse = mse
        
        best_split = (possible_splits[index], best_mse)
            
        return best_split

    def impurity(self, split, data):
        """
        The impurity of a split is:
        (number of samples in the left node / number of samples in the parent node) *
        mse(left node) + (number of samples in the right node / number of samples in
        the parent node) * mse(right node).

        split[0]: question_type {"<=", "in"}
        split[1]: attribute_index
        split[2]: split {set of attributes for categorical ("in" question_type), 
        a number for numerical}
        """
        left_node, right_node = self.split_data(split, data)
        mse_left = self.mean_squared_error(left_node.data)
        mse_right = self.mean_squared_error(right_node.data)
        
        return (len(left_node.data.data) / len(data.data)) * mse_left + (len(right_node.data.data) / len(data.data)) * mse_right

    def split_data(self, split, data):
        data_copy = deepcopy(data)
        data_copy.data = []
        # true node
        left_node = Node(data = data_copy)
        data_copy = deepcopy(data_copy)
        # false node
        right_node = Node(data = data_copy)
        
        for instance in data.data:
            if split[0] == '<=':
                if instance[split[1]] <= split[2]:
                    left_node.data.data.append(instance)
                else:
                    right_node.data.data.append(instance)
            else: # split[0] == 'in'
                if instance[split[1]] in split[2]:
                    left_node.data.data.append(instance)
                else:
                    right_node.data.data.append(instance)  
        return (left_node, right_node)

    def mean_squared_error(self, data):
        #print(len(data.data))
        if len(data.data) == 0:
            return 0
        target_class_index = data.attributes.index(self.target_class)
        mean = 0
        mse = 0
        
        for i, instance in enumerate(data.data):
            mean += (1.0 / (i+1)) * (instance[target_class_index] - mean)
        for instance in data.data:
            mse += (instance[target_class_index] - mean)**2
            
        
        mse = mse / len(data.data)
        return mse

    def generate_candidate_splits(self, attribute_index, data):
        """
        Generates all the possible splits for a given attribute (attribute type can be 
        numeric or categorical)

        If the attribute corresponding to the attribute_index is categorical then
        return of type: ("in", [sets of splits (e.g. ['5'], ['5', '2'])])

        If the attribute corresponding to the attribute_index is numeric then
        retun of type: ("<=", set of splits (e.g. 2, 3.5, 7))
        """
        question_type = '<='
        splits = []
        if data.attribute_types[attribute_index] == 'categorical':
            question_type = 'in'
            splits = self.create_categorical_split(data, attribute_index)
        else: # numeric
            splits = self.create_numeric_split(data, attribute_index)
        return (question_type, splits)
        
    def create_numeric_split(self, data, attribute_index):
        """
        Numeric splits are created by first sorting the values of the attributes,
        then taking the averages of two neighbouring values.
        e.g. a split is (sorted_values[0] + sorted_values[1])/2

        Modification: Uniform approximation
        To speed up the algorithm we assume that the values are distributed uniformly
        between the minimum and maximum value, and we select k split points the following way:
        split[i] = min + i*(max-min)/(k+1)
        """
        
        #splits = []
        #values = [instance[attribute_index] for instance in data.data]
        #values.sort()
        #for i in range(len(values)-1):
        #    split = (values[i] + values[i+1]) / 2
        #    if split not in splits:
        #        splits.append(split)
        #uniform approximation
        k = 10
        max = data.data[0][attribute_index]
        min = data.data[0][attribute_index]
        for instance in data.data:
            if instance[attribute_index] < min:
                min = instance[attribute_index]
            if instance[attribute_index] > max:
                max = instance[attribute_index]
        splits=[]
        for i in range(k):
            #print(i)
            splits.append(min + (i+1)*(max-min) / (k+1))
        
        return splits

    
    def create_categorical_split(self, data, attribute_index):
        """
        The categories are ordered by increasing mean of the outcome (Assign for each
        category the mean of the outcomes from that category and sort the categories 
        increasingly). Then the optimal split is one of the k-1 splits of the ordered
        categories. This reduces the complexity from 2^(k) to k, where k is the 
        number of categories. 
        """
        categories = data.attribute_values[attribute_index]
        target_class_index = data.attributes.index(self.target_class)
        means = []
        for category in categories:
            values = [instance[target_class_index] for instance in data.data if instance[attribute_index] == category]
            if len(values) == 0:
                continue
            means.append((category, sum(values) / len(values)))

        sorted_categories = sorted(means, key=lambda category: category[1])
        splits = []
        for i in range(len(sorted_categories)-1):
            splits.append([category[0] for category in sorted_categories[:i+1]])
        return splits



class Node:

    def __init__(self, is_leaf = False, data = None, parent = None, left_child = None, right_child = None):
        self.is_leaf = is_leaf
        self.data = data
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.split = None