import ast

class Data:
    def __init__(self, file=None):
        self.relation = None
        self.attributes = None
        self.attribute_types = None
        self.attribute_values = None
        self.n_attributes = None
        self.data = None

        if file is not None:
            self.load_arff(file)

        #self.data = {}

    def load_arff(self, file):
        # arff data object
        #data = {}

        # separator in file
        delim = ' '

        line = file.readline()
        row = line.strip().split()
        
        if row[0] != '@relation':
            print("No relation name found!")
            exit(1)

        self.relation = row[1]
        #data['relation'] = row[1]
        self.attributes = []
        #data['attributes'] = []
        self.attribute_types = []
        #data['attribute_types'] = []
        self.attribute_values = []
        #data['attribute_values'] = []
        self.n_attributes = 0
        #data['attribute_number'] = 0
        self.data = []

        line = file.readline()
        while line:
            row = line.strip().split(delim)
            if row[0] == '@attribute':
                self.n_attributes += 1
                self.attributes.append(row[1])
                if row[2] == 'numeric':
                    self.attribute_types.append(row[2])
                    self.attribute_values.append("numeric")
                else:
                    self.attribute_types.append("categorical")
                    categories = row[2].strip("{").strip("}").split(',')
                    self.attribute_values.append(categories)
            elif row[0] == '@data':
                delim = ','
            else:
                for i, instance in enumerate(row):
                    if self.attribute_types[i] == 'numeric':
                        row[i] = ast.literal_eval(row[i])
                self.data.append(row)
            line = file.readline()
        #self.data = data

    def summary(self):
        print("---------------- " + self.relation + " ----------------")
        for attribute in self.attributes:
            print(attribute + " | ", end="")
        print()
        for data in self.data:
            for i in range(len(data)):
                print(str(data[i]) + " | ", end="")
            print()
