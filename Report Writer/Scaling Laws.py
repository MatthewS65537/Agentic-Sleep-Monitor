import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sizes = [0.5, 1.5, 3, 7, 14]
One_Pass = [4.64, 5.88, 5.54, 6.98, 7.06]
MoA_Best = [5.19, 6.84, 7.56, 7.88, 7.99]

def linear_func(x, a, b):
    return a * x + b

# Log-transform the sizes
log_sizes = np.log(sizes)

# Fit linear regression on log-transformed data
popt_one_pass, _ = curve_fit(linear_func, log_sizes, One_Pass)
popt_moa_best, _ = curve_fit(linear_func, log_sizes, MoA_Best)

# Print fitted parameters
print("One Pass fit: y = {:.2f} * log(x) + {:.2f}".format(*popt_one_pass))
print("MoA Best fit: y = {:.2f} * log(x) + {:.2f}".format(*popt_moa_best))

# Generate points for smooth curves
x_smooth = np.linspace(np.log(min(sizes)), np.log(max(sizes)), 100)
y_one_pass = linear_func(x_smooth, *popt_one_pass)
y_moa_best = linear_func(x_smooth, *popt_moa_best)

plt.figure(figsize=(10, 6))
plt.scatter(sizes, One_Pass, label='One Pass', color='blue')
plt.scatter(sizes, MoA_Best, label='MoA Best', color='red')
plt.plot(np.exp(x_smooth), y_one_pass, '--', color='blue')
plt.plot(np.exp(x_smooth), y_moa_best, '--', color='red')

plt.xlabel('Model Size (Billion Parameters)')
plt.ylabel('Performance Score')
plt.title('Model Performance vs Size')
plt.legend()
# plt.xscale('log')
plt.grid(True)

plt.savefig("./graphs/scaling_laws_Qwen2.5.png")