import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sizes = [0.5, 1.5, 3, 7, 14, 32]
One_Pass = [4.64, 5.88, 5.54, 6.98, 7.06, 6.89]
MoA_Best = [5.19, 6.84, 7.56, 7.88, 7.99, 7.96]

def log_func(x, a, b):
    return np.log(a * np.log(x) + b)

# Fit logarithmic regression
popt_one_pass, _ = curve_fit(log_func, sizes, One_Pass, maxfev=10000)
popt_moa_best, _ = curve_fit(log_func, sizes, MoA_Best, maxfev=10000)

# Print fitted parameters
print("One Pass fit: y = log({:.2f} * log(x) + {:.2f})".format(*popt_one_pass))
print("MoA Best fit: y = log({:.2f} * log(x) + {:.2f})".format(*popt_moa_best))

# Generate points for smooth curves
x_smooth = np.logspace(-1, np.log10(50), 1000)
y_one_pass = log_func(x_smooth, *popt_one_pass)
y_moa_best = log_func(x_smooth, *popt_moa_best)

plt.figure(figsize=(10, 6))
plt.scatter(sizes, One_Pass, label='One Pass', color='blue')
plt.scatter(sizes, MoA_Best, label='MoA Best', color='red')
plt.plot(x_smooth, y_one_pass, '--', color='blue')
plt.plot(x_smooth, y_moa_best, '--', color='red')

# Add SOTA line
plt.axhline(y=8.31, color='green', linestyle=':', label='SOTA (1Pass)')
plt.axhline(y=8.34, color='purple', linestyle=':', label='SOTA (MoA)')

plt.xlabel('Model Size (Billion Parameters, Log Scale)', fontsize=14)
plt.ylabel('Performance Score', fontsize=14)
plt.title('Qwen2.5 Instruct Model Family Performance vs Model Size (Log Scale)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Set x-axis to log scale with appropriate labels
plt.xscale('log')
plt.xticks([1, 5, 10, 25, 50], ['1', '5', '10', '25', '50'])

# Adjust y-axis to include SOTA line
plt.ylim(4, 8.5)

# Add equations to the plot
equation_one_pass = f"One Pass: y = log({popt_one_pass[0]:.2f} * log(x) + {popt_one_pass[1]:.2f})"
equation_moa_best = f"MoA Best: y = log({popt_moa_best[0]:.2f} * log(x) + {popt_moa_best[1]:.2f})"
plt.text(0.05, 0.92, equation_one_pass + '\n' + equation_moa_best, transform=plt.gca().transAxes, 
         verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.savefig("./graphs/scaling_laws_Qwen2.5.png") #DON'T CHANGE THIS
