import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("food_truck_data.txt", delimiter=",")
X = data[:, 0]
y = data[:, 1]

def predict(x, w, b):
    return w * x + b

w, b = 1, -3
y_pred = predict(X, w, b)

plt.scatter(X, y, marker="x", c="r")
plt.plot(X, y_pred, c="b")
plt.xlabel("Population (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.show()

new_cities = [6.5, 3.0]
for city in new_cities:
    profit = predict(city, w, b)
    if profit > 0:
        print(f"Open in city with population {city*10000:.0f}: Expected profit {profit*10000:.2f}")
    else:
        print(f"Do NOT open in city with population {city*10000:.0f}: Expected loss {profit*10000:.2f}")
