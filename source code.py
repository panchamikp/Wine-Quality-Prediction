import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:\\Users\\user\\Desktop\\datset\\winequality-red.csv')

# Convert the regression problem to a classification problem. 
# Note: Adjust the threshold as necessary. Here, wines with a quality of 7 or above are considered 'good'.
data['good_quality'] = data['quality'] >= 7
data['good_quality'] = data['good_quality'].astype(int)

# Splitting the dataset
X = data.drop(columns=['quality', 'good_quality'])
y = data['good_quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Set up the grid search parameters
param_grid = {
    'clf__n_estimators': [50, 100, 150],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from grid search
print("Best Parameters: ", grid_search.best_params_)

# Cross-validation with optimized parameters
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print("Average Cross-validation Score: ", cv_scores.mean())

# Predict using the best model
y_pred = grid_search.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
