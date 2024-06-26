import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the Titanic dataset from the provided file path
train_data_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\archive\\train.csv'
train_data = pd.read_csv(train_data_path)

# Split the train data into features and target variable
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing for numeric and categorical features
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [10, 15],
    'classifier__min_samples_leaf': [5, 10],
    'classifier__bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train with GridSearchCV
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Beste Parameter:", grid_search.best_params_)

# Create the best model
best_clf = grid_search.best_estimator_

# Calculate and print accuracy for train and test sets
train_accuracy = best_clf.score(X_train, y_train)
test_accuracy = best_clf.score(X_test, y_test)

print(f"Train Data Accuracy: {train_accuracy}")
print(f"Test Data Accuracy: {test_accuracy}")



