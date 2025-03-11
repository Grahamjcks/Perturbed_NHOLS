import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data (change filename as needed)
df = pd.read_csv("error_bounds_ls_caltech.csv")
#df = pd.read_csv("error_bounds_nhols_caltech.csv")

# Extract x and y values from the CSV
# x-axis: error_matrix_L2, y-axis: convergent_solution_L2_difference
x = df["error_matrix_L2"].values
y = df["convergent_solution_L2_difference"].values

# Compute the unconstrained best-fit line (linear regression) using np.polyfit for degree 1
coeffs = np.polyfit(x, y, 1)  # returns [slope, intercept]
m, b = coeffs
fit_fn = np.poly1d(coeffs)  # best-fit function: y = m*x + b

# Compute the R^2 score
y_pred = m * x + b
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

# Create a smooth range of x values for plotting the fitted line
x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = m * x_fit + b  # best-fit line

# Define the bound line: the identity line y = x
y_bound = x_fit

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the data points as a scatter plot
plt.scatter(x, y, color="blue", label="Points denoting different levels of perturbation")

# Plot the best-fit line (red, dotted) and fill the region under it
plt.plot(x_fit, y_fit, linestyle="--", color="red",
         label=f"$\\|\\tilde{{F}}-F\\| = {m:.2f}\\,\\|Z\\| + {b:.2f}$, $R^2 = {r2:.2f}$")
plt.fill_between(x_fit, 0, y_fit, color="red", alpha=0.1)

# Plot the bound line (green, dotted) and fill the region under it
plt.plot(x_fit, y_bound, linestyle=":", color="green",
         label="$\\|\\tilde{F}^*-F^*\\| \\leq \\|Z\\|$")
plt.fill_between(x_fit, 0, y_bound, color="green", alpha=0.2)

# Label the axes and add title, legend, and grid
plt.xlabel(f"$L_2$ Norm of Error Matrix")
plt.ylabel("$L_2$ Norm of Solution Difference")
plt.title(f"Error Matrix vs. Solution Difference $L_2$ Norms with Best-Fit Line:\n Standard LS on Caltech36 Dataset (30% labeled data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()