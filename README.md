# Diabetes Prediction System – ML & Streamlit App

An end-to-end machine learning project designed to predict whether a person is diabetic based on key medical attributes. This project covers the complete data science workflow — from data analysis and model training to deployment as an interactive web application using Streamlit.

---

## Project Overview

The objective of this project is to build a reliable classification model that can predict the likelihood of diabetes using health-related features such as:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI
* Diabetes Pedigree Function
* Age

The final model is deployed as a web app using **Streamlit**, allowing users to input health parameters and receive an instant prediction.

---

## Exploratory Data Analysis (EDA)

* Analyzed data distribution and feature relationships
* Identified missing values and outliers
* Used visualizations (histograms, box plots, heatmaps) for insights
* Checked correlation between features and the target variable

---

## Data Preprocessing

* Handled missing/zero values in key columns
* Applied feature scaling (StandardScaler )
* Split the dataset into training and testing sets

---

## Model Building

Trained and evaluated multiple machine learning models:

* Logistic Regression
* Random Forest

Models were compared using:

* Accuracy
Pending:
* Precision
* Recall
* F1-Score
* Confusion Matrix

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) was performed to improve performance.

---

## Deployment

The best-performing model was deployed using **Streamlit** with a simple and interactive UI.
Users can enter their medical data and instantly get diabetes prediction results.

---

## Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Streamlit

---

## How to Run the Project

1. Clone the repository:
   git clone [your-github-repo-link]

2. Install the required libraries:
   pip install -r requirements.txt

3. Run the Streamlit App:
   streamlit run app.py

4. Open the app in your browser and try your own inputs.

---

## Results

The selected model achieved strong performance with good balance between precision and recall, making it suitable for real-world predictions.

---

## Future Improvements

* Add more medical features
* Improve model accuracy using deep learning
* Add data upload functionality
  

---

⭐ If you found this project helpful, feel free to star the repository!
