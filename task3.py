import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("./Task3/titanic/train.csv")
test_data = pd.read_csv("./Task3/titanic/test.csv")
columns_to_drop = ["Name", "Ticket", "Cabin", "Embarked"]
train_data_clean = train_data.drop(columns_to_drop, axis=1)
test_data_clean = test_data.drop(columns_to_drop, axis=1)
le = LabelEncoder()
train_data_clean["Sex"] = le.fit_transform(train_data_clean["Sex"])
test_data_clean["Sex"] = le.fit_transform(test_data_clean["Sex"])
train_data_clean = train_data_clean.fillna(train_data_clean["Age"].mean())
# train_data_clean = train_data_clean.fillna(train_data_clean["Fare"].mean())
test_data_clean = test_data_clean.fillna(test_data_clean["Age"].mean())
# test_data_clean = test_data_clean.fillna(test_data_clean["Fare"].mean())
input_cols = ['Pclass', "Sex", "Age", "SibSp", "Parch", "Fare"]
output_cols = ["Survived"]
X_train = np.array(train_data_clean[input_cols])
Y_train = np.array(train_data_clean[output_cols])
X_test = np.array(test_data_clean[input_cols])
Test_data_classes = pd.read_csv("./Task3/titanic/Answer.csv")
Test_data_classes = Test_data_classes["Survived"]

# best_agerange=0
# max_score = 0
# bestnodenum = 0
# bestdepth = 0

X_train[:, input_cols.index("Age")] = np.array(X_train[:, input_cols.index("Age")] / 10, dtype=int)
X_test[:, input_cols.index("Age")] = np.array(X_test[:, input_cols.index("Age")] / 10, dtype=int)


decision_tree_classifier = DecisionTreeClassifier(max_leaf_nodes=16)
decision_tree_classifier.fit(X_train, Y_train)
decision_tree_output = decision_tree_classifier.predict(X_test)

# df = pd.DataFrame({"PassengerId": test_data_clean["PassengerId"], "Survived": decision_tree_output})
# df.to_csv('./Task3/titanic/result.csv', index=False)

score = accuracy_score(Test_data_classes, decision_tree_output)
print(score)
# if score > max_score:
#     max_score = score
#     best_agerange = agerange
#     bestdepth = depth
#     bestnodenum = numofnodes
# print(best_agerange,bestnodenum, bestdepth,max_score)

plt.figure(figsize=(50, 10))
p = plot_tree(decision_tree_classifier, feature_names=input_cols, class_names=["0", "1"], filled=True, rounded=True,
              fontsize=14)
plt.savefig("./Task3/titanic/tree2.jpg")
plt.show()
