# Depoying Linear Regresssion model

# Importing the basic packages
import pandas as pd

# Importing the Model Packages
from sklearn.model_selection import train_test_split

#To bring the data(independent variables) into same scale
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Read the csv file
diabetic_data = pd.read_csv(url, names=columns)

# Splilt the data x and y
x = diabetic_data.iloc[:,0:8]
y = diabetic_data.iloc[:,8]

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=101)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model 
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

print(f'[INFO] model training is completed')
result = model.score(x_test_scaled, y_test)
print(f'[INFO] Test Score is {result}')

# Importing Serialization Packages
import joblib 
joblib.dump(model, 'model_scaled.pkl') # Converting model to pickle format
joblib.dump(scaler, 'scaled.pkl') # Converting scaling parameter to pickle format