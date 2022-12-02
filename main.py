# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

# Load Data
attribute_names = ['date', 'home_team', 'away_team', 'home_team_continent', 'away_team_continent', 'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'home_team_score', 'away_team_score', 'tournament', 'city', 'country', 'neutral_location',
                   'shoot_out', 'home_team_result', 'home_team_goalkeeper_score', 'away_team_goalkeeper_score', 'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score', 'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score']

# attribute_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
fifa_data = pd.read_csv("international_matches.csv", names=attribute_names)


# split dataset in features and target variable
# feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
feature_cols = ['date', 'home_team', 'away_team', 'home_team_continent', 'away_team_continent', 'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'home_team_score', 'away_team_score', 'tournament', 'city', 'country', 'neutral_location',
                'shoot_out', 'home_team_result', 'home_team_goalkeeper_score', 'away_team_goalkeeper_score', 'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score', 'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score']

X = fifa_data[feature_cols]  # Features
y = fifa_data.home_team_result  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


# CHANGE HERE FOR different  criterion, max_depth, min_samples_leaf
# Create Decision Tree classifer object
# #Try for entropy/gini
# criterion='entropy', max_depth=5,min_samples_leaf=5
clf = DecisionTreeClassifier(
    criterion='gini', max_depth=8, min_samples_leaf=14)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Accuracy on whole test set
acc = metrics.accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", acc)


#########################################################
# Visualize the Tree

attribute_names = ['date', 'home_team', 'away_team', 'home_team_continent', 'away_team_continent', 'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'home_team_score', 'away_team_score', 'tournament', 'city', 'country', 'neutral_location',
                   'shoot_out', 'home_team_result', 'home_team_goalkeeper_score', 'away_team_goalkeeper_score', 'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score', 'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score']
class_name = ['win', '0', '-1']

# Fix your figure size here
# dpi=dots per pixel
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(100, 100), dpi=10)
# tree.plot_tree(clf,
#                feature_names=attribute_names,
#                class_names=class_name,
#                filled=True)
# fig.savefig('decissionTree.png')
