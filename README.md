# Iris Flower Classification 🌸

## OIBSIP Data Science Internship - Task 1

## Project Overview
This project focuses on building a Machine Learning classification model to identify the species of an Iris flower based on its physical measurements.

The model predicts three different Iris species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

The project includes data analysis, visualization, model training, and performance evaluation using different classification algorithms.

## Objective
To develop a machine learning model that can accurately classify Iris flower species using features such as:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset
The Iris dataset is obtained from Scikit-learn's built-in dataset library.

Dataset contains:
- 150 samples
- 4 numerical features
- 3 target classes

## Project Workflow

### 1. Data Loading
- Loaded Iris dataset using Scikit-learn.
- Converted data into a Pandas DataFrame.

### 2. Exploratory Data Analysis (EDA)
Performed:
- Dataset shape analysis
- Data type checking
- Null value checking
- Statistical summary
- Feature distribution analysis

### 3. Data Visualization
Created visualizations:
- Pairplot to understand feature relationships
- Scatter plots for species separation
- Box plots for feature distribution

### 4. Model Training
Trained multiple classification models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

### 5. Model Evaluation
Evaluated models using:
- Accuracy Score
- Confusion Matrix
- Classification Report
  - Precision
  - Recall
  - F1-score

## Results
The models were compared based on their performance, and the best-performing model was selected for Iris flower classification.

## Project Structure




---------------------------------------------------------------------------------------------------
## Car Price Prediction with Machine Learning
Predicts the selling price of used cars using regression models trained on real listing data (car_data.xls).

Overview
This project builds an end-to-end ML pipeline that:

Loads and cleans the car listings dataset
Engineers new features (car age, brand)
Explores the data visually (distributions, correlations)
Trains and compares three regression models
Picks the best-performing model
Reports which features matter most
Predicts the price of a sample car
Dataset
car_data.xls (CSV format) — 301 used car listings with the following columns:

Column	Description
Car_Name	Model name of the car
Year	Year of manufacture
Selling_Price	Price the car was sold for (in lakhs)
Present_Price	Current showroom/ex-showroom price (in lakhs)
Kms_Driven	Total kilometers driven
Fuel_Type	Petrol / Diesel / CNG
Seller_Type	Dealer / Individual
Transmission	Manual / Automatic
Owner	Number of previous owners
Files
.
├── car_price_prediction.py   # main script
├── car_data.xls               # dataset
└── plots/                     # generated charts (created on run)
Requirements
Python 3.9+
pandas, numpy, matplotlib, seaborn, scikit-learn
Install dependencies:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Run the script from the same folder as car_data.xls:

bash
python car_price_prediction.py
The script prints progress and metrics to the console and saves all charts into a plots/ folder.

Pipeline Steps
Load data — reads car_data.xls
Clean data — removes duplicates, normalizes column names, fills missing values
Clean categoricals — lowercases and trims fuel_type, seller_type, transmission
Feature engineering
car_age = 2026 - Year
brand = first word of Car_Name
EDA (saved as plots)
Selling price distribution
Price vs fuel type
Price vs car age
Correlation heatmap
Preprocessing — one-hot encodes categorical features
Modeling — trains 3 models on an 80/20 train/test split:
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
Evaluation — compares models using MAE, RMSE, and R²
Feature importance — from the Random Forest model
Sample prediction — predicts the price of an example car
Results
On this dataset, Linear Regression was the best-performing model:

Model	MAE	RMSE	R² Score
Linear Regression	1.35	2.39	0.78
Gradient Boosting	1.12	2.60	0.74
Random Forest	1.37	3.30	0.58
Present_Price is by far the strongest predictor of selling price (~89% feature importance), followed by car_age.

Notes
The original version of this project downloaded a different, external dataset from GitHub instead of using the provided car_data.xls. This version was corrected to load and use the given file directly, so all cleaning, analysis, and model results reflect the actual data supplied.



