import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv("Datasets/main.csv")  # Replace with the correct file path

# Step 2: Analyze real-life depression level percentages
real_counts = data['depressionLevel'].value_counts()
real_percentages = (real_counts / len(data)) * 100

# Display the real-life percentages
print("Real-Life Depression Level Percentages:")
print(real_percentages)

# Step 3: Prepare the data for prediction
x = data.drop(columns=['depressionLevel'])  # Features
y = data['depressionLevel']  # Target

# Encode categorical target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.4, random_state=25)

# Train SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(x_train, y_train)

# Generate predictions
y_pred = svm_model.predict(x_test)

# Step 4: Analyze predicted percentages
predicted_counts = pd.Series(y_pred).value_counts()
predicted_percentages = (predicted_counts / len(y_pred)) * 100

# Match predicted classes with their labels
predicted_classes = le.inverse_transform(predicted_counts.index)
predicted_percentages_dict = dict(zip(predicted_classes, predicted_percentages))

# Step 5: Realistic Depression Level Percentages
realistic_percentages = {
    "Low": 40,
    "Mild": 30,
    "Moderate": 20,
    "High": 10
}

# Combine data into a DataFrame for comparison
all_levels = ['Low', 'Mild', 'Moderate', 'High']
comparison_df = pd.DataFrame({
    "Real-Life": [real_percentages.get(level, 0) for level in all_levels],
    "Predicted": [predicted_percentages_dict.get(level, 0) for level in all_levels]
}, index=all_levels)

print("\nComparison of Distributions:")
print(comparison_df)

# Step 6: Plot comparison bar chart
comparison_df.plot(kind='bar', figsize=(10, 6), color=['lightblue', 'orange', 'green'])
plt.title("Comparison of Depression Level Distributions")
plt.ylabel("Percentage (%)")
plt.xlabel("Depression Levels")
plt.legend(title="Distribution Source")
plt.tight_layout()
plt.show()
