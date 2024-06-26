import pandas as pd

# Load the Titanic dataset from the provided file path
train_data_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\archive\\train.csv'
train_data = pd.read_csv(train_data_path)

# Anzeigen der Anzahl fehlender Werte pro Spalte
missing_values = train_data.isnull().sum()
print("Anzahl fehlender Werte pro Spalte:\n", missing_values)

import pandas as pd

# Load the Titanic dataset from the provided file path
train_data_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\archive\\train.csv'
train_data = pd.read_csv(train_data_path)

# Anzeigen der Anzahl fehlender Werte in den Spalten 'Age' und 'Embarked'
missing_age = train_data['Age'].isnull().sum()
missing_embarked = train_data['Embarked'].isnull().sum()

print(f"Number of missing values in the 'Age' column': {missing_age}")
print(f"Number of missing values in the 'Embarked' column: {missing_embarked}")