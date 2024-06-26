import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Lade den Titanic-Datensatz
train_data_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\archive\\train.csv'
train_data = pd.read_csv(train_data_path)

# Datenvorverarbeitung für den Trainingsdatensatz
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())

# Entfernen unnötiger Spalten
train_data.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Umwandlung kategorialer Variablen in numerische
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# Aufteilen der Daten in Features und Zielvariable
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definieren des Parameterraums für GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialisieren des RandomForestClassifiers
clf = RandomForestClassifier(random_state=42)

# Initialisieren von GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Training mit GridSearchCV
grid_search.fit(X_train, y_train)

# Ausgabe der besten Parameter
print("Beste Parameter:", grid_search.best_params_)