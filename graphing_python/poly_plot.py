import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Read CSV data (ensure the file is in the correct path)
df = pd.read_csv("perturbation_analysis3.csv")

# Sort data by perturbed_standard_deviation to maintain order
df.sort_values("perturbed_standard_deviation", inplace=True)

# Extract columns
x = df["perturbed_standard_deviation"].values
y_unperturbed = df["unperturbed_accuracy"].values
y_perturbed = df["perturbed_accuracy"].values

# Set the polynomial degree (adjust if needed)
poly_degree = 3

# Fit a polynomial to the unperturbed and perturbed data
coeffs_unperturbed = np.polyfit(x, y_unperturbed, poly_degree)
coeffs_perturbed = np.polyfit(x, y_perturbed, poly_degree)

# Create polynomial functions from the coefficients
poly_unperturbed = np.poly1d(coeffs_unperturbed)
poly_perturbed = np.poly1d(coeffs_perturbed)

# Create a smooth range of x values for plotting the fitted curves
x_new = np.linspace(x.min(), x.max(), 300)
y_unperturbed_poly = poly_unperturbed(x_new)
y_perturbed_poly = poly_perturbed(x_new)

# Calculate the fixed error height from the unperturbed accuracy range
error_height = np.max(y_unperturbed) - np.min(y_unperturbed)

# Define the error tube boundaries: a flat band starting at the unperturbed polynomial fit
lower_bound = y_unperturbed_poly
upper_bound = y_unperturbed_poly + error_height

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the polynomial fit for perturbed accuracy
plt.plot(x_new, y_perturbed_poly, label="Perturbation Label Accuracy (n=3 Poly Fit)", color="orange")

# Plot the original data points
plt.scatter(x, y_unperturbed, color="blue", s=50, zorder=5, label="Unmodified Data")
plt.scatter(x, y_perturbed, color="orange", s=50, zorder=5, label="Perturbation Label Data")

# Draw vertical lines connecting corresponding unperturbed and perturbed points
for xi, y_unp, y_per in zip(x, y_unperturbed, y_perturbed):
    plt.plot([xi, xi], [y_unp, y_per], color="gray", linestyle="--", zorder=3)

# Optionally, plot the error tube as a shaded region directly on top of the unperturbed fit
# plt.fill_between(x_new, lower_bound, upper_bound, color="blue", alpha=0.2, label="Error Tube")

# Add a red mark at x = 0.74 on the perturbed polynomial curve
red_x = 0.74
red_y = poly_perturbed(red_x)
plt.scatter([red_x], [red_y], color="red", s=100, zorder=10, label="Around half of values have over 50% perturbation")

# Add labels, title, legend, and grid
plt.xlabel("Perturbation Standard Deviation")
plt.ylabel("Percentage Labeling Accuracy")
plt.title("20% Labeled: Labeling Accuracy vs. Perturbation Standard Deviation")
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

# Zoom out on the plot so that the y-axis goes from 0 to 100%
plt.gca().set_ylim(bottom=0, top=100)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()