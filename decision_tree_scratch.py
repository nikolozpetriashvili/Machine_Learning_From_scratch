from __future__ import print_function

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]

#1.mtliani monacemebis analizi da avarchiot kvelaze kargi shekitxva
#2.shemdgomshi davkot ise rogorc information gains
#3.dakofili monacemebit vadgen titoeulis ginis. tu romelime 0-s agar vkoft...

#1.gvchirdeb funqcia romelic itvlis gini-s
#2.gvchirdeba funqcia romelic itvlis information gain-s da weighted avg gini-s
#3.gvchirdeba funqcia romelic klasebis mixedvit dakofs awmyoshi arsebul monacemebs
#4.avarchiot kvelaze kargi shekitxva info gain-s meshveobit,romelic dayofs xes nawilebad
#5.dagvchirdeba shekitxvis klasi romelic sheinaxavs shekitxvas da aseve dagvexmareba monacemebis true da false-ad dayofashi
#6.true false-ad dayofistvis gvchirdeb funqcia romelic daadgens monacemi ricxvia tu string

def is_numeric(value):
    return isinstance(value,int) or isinstance(value,float)

class Question:
    def __init__(self,col,val):
        self.col = col
        self.val = val
    
    def match(self,row):
        value = row[self.col]
        if is_numeric(value):
            return value >= self.val
        else:
            return value == self.val
        
    def __repr__(self):
        condition = "=="
        if is_numeric(self.val):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.col], condition, str(self.val))

def count_lbl(data):
    hash_map = {}
    for row in data:
        lbl = row[-1]
        if lbl not in hash_map:
            hash_map[lbl] = 0
        hash_map[lbl] += 1

    return hash_map

def calc_gini(data):
    labels = count_lbl(data) 
    impurity = 1

    for label in labels:
        impurity -= (labels[label]/len(data))**2

    return impurity

def partition_data(rows,question):
    true_rows,false_rows = [],[]

    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)

    return true_rows,false_rows

def info_gain(left,right,current_impurity):
    p = float(len(left)/(len(left) + len(right)))
    return current_impurity - p * calc_gini(left) - (1-p) * calc_gini(right)

def find_best_split(data):
    best_gain = 0
    best_question = None
    current_impurity = calc_gini(data)
    n_features = len(data[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in data])
        for val in values:
            question = Question(col,val)
            true_rows , false_rows = partition_data(data,question)   

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows,false_rows,current_impurity)

            if gain > best_gain:
                best_gain , best_question = gain , question
    return best_gain,best_question

class Leaf:
    def __init__(self,rows):
        self.predictions = count_lbl(rows)
    
class Decision_Node:
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain,question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    
    true_rows,false_rows = partition_data(rows,question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question,true_branch,false_branch)

def print_tree(node,spacing=''):
    
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

my_tree = build_tree(training_data)
print_tree(my_tree)

def classify(row, node):
    if isinstance(node, Leaf):
        print(node.predictions)
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

classify(training_data[0], my_tree)