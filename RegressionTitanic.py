import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\archive\\train.csv'
titanic_data = pd.read_csv(data_path)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


# Definieren der Zielvariable und Merkmale
X = titanic_data.drop(columns=['Age'])
y = titanic_data['Age']

# Fehlende Werte in Age ausfüllen
data = titanic_data[['Age']].dropna()
X = X.loc[data.index]
y = data['Age']

# Aufteilen der Daten in Trainings- und Testsets, Random State 42 for fun
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline für numerische Daten
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline für kategorische Daten: Fehlende Werte durch den häufigsten Wert ersetzen und One-Hot-Encoding anwenden
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Kategorische und numerische Merkmale identifizieren
categorical_features = titanic_data.select_dtypes(include=['object']).columns
numerical_features = titanic_data.select_dtypes(include=['number']).columns.drop('Age')

# Kombinieren der numerischen und kategorischen Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Funktion zum Erstellen und Trainieren eines Modells
def build_pipeline(model):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{model.__class__.__name__} Mean Squared Error:', mse)
    print(f'{model.__class__.__name__} Predictions:', y_pred[:5])  # Zeige die ersten 5 Vorhersagen
    print(f'{model.__class__.__name__} Actual:', y_test.values[:5])  # Anzeigen der ersten 5 Werte
    return pipeline

from sklearn.linear_model import LinearRegression

# Lineare Regression
linear_model = build_pipeline(LinearRegression())
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

from sklearn.tree import DecisionTreeRegressor

# Entscheidungsbaumregression
tree_model = build_pipeline(DecisionTreeRegressor(random_state=42))
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)

from sklearn.ensemble import RandomForestRegressor

# Random Forest Regression
forest_model = build_pipeline(RandomForestRegressor(random_state=42, n_estimators=100))
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)

# Ergebnisse visualisieren
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
mse_values = [mse_linear, mse_tree, mse_forest]

plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('MSE Vergleich')
plt.show()


#Interpretation: Ein niedriger MSE zeigt an, dass die Vorhersagen des Modells im Durchschnitt nah an den tatsächlichen Werten liegen.
#Ein höherer MSE deutet darauf hin, dass das Modell größere Abweichungen in den Vorhersagen hat.