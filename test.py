import joblib
import numpy as np

# Load the saved model
model = joblib.load('model_scaled.pkl')
scaler = joblib.load('scaled.pkl')

# user_input = np.array([[1,2,3,4,5,6,7,8]])
user_input = np.array([[6,148,72,35,0,33.6,0.627,50]])

scaled_input = scaler.transform(user_input)

# Model Prediction
result = model.predict(user_input)

print('[INFO] The result is ', result[0])

