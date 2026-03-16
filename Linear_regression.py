import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.array([[1],[2],[3]])
y=np.array([[10],[20],[30]])

lr=LinearRegression().fit(x,y)
X_test=np.linspace(0,4,20).reshape(-1,1)
y_predict = lr.predict(X_test)

plt.scatter(x, y, label="training data")
plt.plot(X_test, y_predict, label="prediction line")
plt.xlim(0, 4)
plt.ylim(0, 40)
plt.title("Predict result")
plt.grid()
plt.legend()
plt.show()
print(lr.coef_)
print(lr.intercept_)


