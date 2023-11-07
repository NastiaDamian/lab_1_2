import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Дані
x = np.array([3.7, 4.1, 5.6, 6.8, 7.9, 8.6, 9.6, 12.3, 11.9, 14.8]).reshape((-1, 1))
y = np.array([15.8, 17.8, 19.8, 23.5, 23.8, 23.8, 26.1, 27.8, 29.1, 36.2])

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept: {new_model.intercept_}")


print(f"slope: {new_model.coef_}")

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")

x_new = np.arange(5).reshape((-1, 1))
x_new
y_new = model.predict(x_new)
y_new

# Графік
plt.scatter(x, y, color='b', label='Дані спостереження')
plt.plot(x, y_pred, color='r', linewidth=2, label='Лінійна регресія')
plt.xlabel('x (незалежна змінна)')
plt.ylabel('y (залежна змінна)')
plt.legend()
plt.show()