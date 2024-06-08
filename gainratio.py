from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create the decision tree classifier
clf = DecisionTreeClassifier(random_state=0, criterion='gini')
clf.fit(X, y)

# Get the feature importances (which include the gain ratio)
importances = clf.feature_importances_

gain_ratios = {}
# Print the gain ratio for each feature
for i, feature in enumerate(iris.feature_names):
    gain = importances[i]
    split = clf.tree_.impurity[i]
    gain_ratio = gain / (split + 1e-7)
    print(f'Gain ratio for {feature}: {gain_ratio:.3f}')

    gain_ratios[feature] = gain_ratio

# Find the feature with the highest gain ratio
best_feature = max(gain_ratios, key=gain_ratios.get)

# Print the best feature for splitting
print(f'Best feature for splitting: {best_feature}')