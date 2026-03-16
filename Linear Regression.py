import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#学習時間
x=np.array([[1],[2],[3]])
#成績があがる
y=np.array([[10],[20],[30]])
lr=LinearRegression().fit(x,y)
#開始値から終了値までを、均等に分けた数を作る,列は1行はNumPyが自動計算
X_test=np.linspace(0,4,20).reshape(-1,1)
print(X_test)
y_predict = lr.predict(X_test)
print(y_predict)
plt.scatter(x, y, label="training data")
plt.plot(X_test, y_predict, label="prediction line")
plt.xlim(0, 4)
plt.ylim(0, 40)
plt.title("Predict result")
plt.grid()
plt.legend()
plt.show()
#傾き（coef）
print(lr.coef_)
#切片（b）
print(lr.intercept_)


