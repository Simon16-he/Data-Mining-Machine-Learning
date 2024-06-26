import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the Titanic dataset from the provided file path
train_data_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\archive\\train.csv'
train_data = pd.read_csv(train_data_path)

# Split the train data into features and target variable
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Sex', 'Embarked', 'Pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model with the best parameters
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42
)

# Create and combine the preprocessing and modeling steps in a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])

# Train the model
pipeline.fit(X_train, y_train)

# Calculate and print accuracy for train and test sets
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print(f"Train Data Accuracy: {train_accuracy}")
print(f"Test Data Accuracy: {test_accuracy}")



