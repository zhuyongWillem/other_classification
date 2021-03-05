# Description：
# Author：朱勇
# Time：2021/3/5 22:47

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

data = pd.read_csv("task1_data.csv")
x = data.drop(["y"],axis=1)
y = data.loc[:,"y"]
model = tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf=50)
model.fit(x,y)
y_predict = model.predict(x)
accuracy = accuracy_score(y,y_predict)
print(accuracy)
x_test = np.array([[1,0,1,1]])
y_test_predict = model.predict(x_test)
print("适合" if y_test_predict == 1 else "不适合")
fig1 = plt.figure(figsize=(200,200))
tree.plot_tree(model,filled="True",feature_names=["Skill","Experience","Degree","Income"],class_names=["Un-qualified","Qualified"])
