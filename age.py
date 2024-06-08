import math
from collections import Counter

# Function to calculate entropy
def entropy(data, target_attribute):
    total = len(data)
    counts = Counter([row[target_attribute] for row in data])
    ent = 0.0
    for count in counts.values():
        prob = count / total
        ent -= prob * math.log2(prob)
    return ent

# Function to calculate information gain
def information_gain(data, target_attribute, attribute):
    total_entropy = entropy(data, target_attribute)
    values = set(row[attribute] for row in data)
    subset_entropy = 0.0
    for value in values:
        subset = [row for row in data if row[attribute] == value]
        prob = len(subset) / len(data)
        subset_entropy += prob * entropy(subset, target_attribute)
    return total_entropy - subset_entropy

# Function to build the decision tree
def build_tree(data, attributes, target_attribute):
    counts = Counter(row[target_attribute] for row in data)
    if len(counts) == 1:
        return list(counts.keys())[0]
    
    if not attributes:
        return counts.most_common(1)[0][0]
    
    gains = [(attribute, information_gain(data, target_attribute, attribute)) for attribute in attributes]
    best_attribute, _ = max(gains, key=lambda x: x[1])
    
    tree = {best_attribute: {}}
    for value in set(row[best_attribute] for row in data):
        subset = [row for row in data if row[best_attribute] == value]
        subtree = build_tree(subset, [attr for attr in attributes if attr != best_attribute], target_attribute)
        tree[best_attribute][value] = subtree
    
    return tree

# Function to print the tree
def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}{key}")
            print_tree(value, indent + "  ")
    else:
        print(f"{indent}{tree}")

# Main program
if __name__ == "__main__":
    # Training data
    data = [
        {'age': '<=30', 'income': 'high', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'no'},
        {'age': '<=30', 'income': 'high', 'student': 'no', 'credit_rating': 'excellent', 'buys_computer': 'no'},
        {'age': '31...40', 'income': 'high', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'yes'},
        {'age': '>40', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'yes'},
        {'age': '>40', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
        {'age': '>40', 'income': 'low', 'student': 'yes', 'credit_rating': 'excellent', 'buys_computer': 'no'},
        {'age': '31...40', 'income': 'low', 'student': 'yes', 'credit_rating': 'excellent', 'buys_computer': 'yes'},
        {'age': '<=30', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'no'},
        {'age': '<=30', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
        {'age': '>40', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
        {'age': '<=30', 'income': 'medium', 'student': 'yes', 'credit_rating': 'excellent', 'buys_computer': 'yes'},
        {'age': '31...40', 'income': 'medium', 'student': 'no', 'credit_rating': 'excellent', 'buys_computer': 'yes'},
        {'age': '31...40', 'income': 'high', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
        {'age': '>40', 'income': 'medium', 'student': 'no', 'credit_rating': 'excellent', 'buys_computer': 'no'}
    ]

    attributes = ['age', 'income', 'student', 'credit_rating']
    target_attribute = 'buys_computer'

    # Build and print the decision tree
    tree = build_tree(data, attributes, target_attribute)
    print_tree(tree)
