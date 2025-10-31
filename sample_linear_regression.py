# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: hours studied vs. test score
hours_studied = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
test_scores = np.array([50, 55, 65, 70, 75, 80])

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(hours_studied, test_scores)

# Predict scores
predicted_scores = model.predict(hours_studied)

# Display the regression line
plt.scatter(hours_studied, test_scores, color='blue', label='Actual Scores')
plt.plot(hours_studied, predicted_scores, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Print the slope and intercept
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
