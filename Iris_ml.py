from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data
iris = load_iris()
X = iris.data
y = iris.target

print("Data loaded. First row:", X[0])  # ✅ This should print something right away

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Data split into training and test sets.")  # ✅

# Choose and train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("Model trained.")  # ✅

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)  # ✅

# Try predicting a custom flower
sample = [[5.1, 3.5, 1.4, 0.2]]
predicted = model.predict(sample)
print("Prediction for sample:", predicted)

