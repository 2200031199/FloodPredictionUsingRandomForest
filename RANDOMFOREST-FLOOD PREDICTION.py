import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data1 = pd.read_csv("Project_train.csv")

# Correlation heatmap
plt.figure(dpi=125)
sns.heatmap(data1.corr(numeric_only=True), annot=True)
plt.show()

# Scatter plot for ClimateChange vs FloodProbability
plt.scatter(data1['ClimateChange'], data1['FloodProbability'])
plt.xlabel('ClimateChange')
plt.ylabel('FloodProbability')
plt.title('Scatter Plot of ClimateChange vs FloodProbability')
plt.show()

# Preparing the data
X = data1[['MonsoonIntensity', 'TopographyDrainage']]
y = data1['FloodProbability']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Flood Probability')
plt.ylabel('Predicted Flood Probability')
plt.title('Random Forest Regression: Actual vs Predicted Flood Probability')
plt.legend()
plt.show()