import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = x_train.shape[0]

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

w = (500.0 - 300.0) / (2.0 - 1.0)
b = 300.0 - (w * 1.0)

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.legend()
plt.show()

x_predict = 1.2
cost_1200sqft = w * x_predict + b

print(f"w: {w}")
print(f"b: {b}")
print(f"Predicted price for a 1200 sqft house: ${cost_1200sqft:,.0f} thousand dollars")