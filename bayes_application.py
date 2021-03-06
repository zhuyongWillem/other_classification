# Description：
# Author：朱勇
# Time：2021/3/6 10:41
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("task2_data.csv")
x = data.drop(["y"],axis=1)
y = data.loc[:,"y"]
model = CategoricalNB()
model.fit(x,y)
y_predict = model.predict(x)
print(accuracy_score(y,y_predict))
x_test = np.array([[2,1,1,1,1],[2,1,1,1,0],[2,1,1,0,0],[2,1,0,0,0],[2,0,0,0,0]])
y_test_predict_prob = model.predict_proba(x_test)
y_test_predict = model.predict(x_test)
#数据组合
test_data_result = np.concatenate((x_test,y_test_predict_prob,y_test_predict.reshape(5,1)),axis=1)
#格式转化
test_data_result = pd.DataFrame(test_data_result)
#列名称替换
test_data_result.columns = ["score","school","award","gender","English","p0","p1","p2","y_test_predict"]
test_data_result.to_csv("test_data_result.csv")