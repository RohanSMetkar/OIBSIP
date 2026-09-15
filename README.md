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




--------------------------------------------------------------------------------------------------
# OIBSIP Data Science Task 2  
# Unemployment Analysis with Python

## 📌 Project Overview

This project is part of the **Oasis Infobyte Data Science Internship Program (OIBSIP)**.

The objective of this project is to perform **Exploratory Data Analysis (EDA)** on unemployment data in India and analyze regional and time-based unemployment trends. The analysis focuses on understanding unemployment patterns and the impact of the **COVID-19 pandemic** on employment conditions.

---

## 🎯 Objective

The main objectives of this project are:

- Analyze unemployment trends in different regions of India.
- Perform data cleaning and preprocessing.
- Explore regional variations in unemployment rates.
- Study monthly unemployment trends.
- Compare unemployment rates of selected regions over time.
- Identify regions with the highest average unemployment rates.
- Analyze relationships between unemployment rate, employment, and labour participation rate.
- Compare pre-COVID and post-COVID unemployment conditions.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📂 Dataset

The dataset contains unemployment-related information from different regions of India.

### Features include:

- Region
- Date
- Estimated Unemployment Rate (%)
- Estimated Employed
- Estimated Labour Participation Rate (%)

---

## 🔍 Project Workflow

### 1. Data Loading
- Imported the unemployment dataset.
- Checked dataset structure and basic information.

### 2. Data Cleaning
Performed preprocessing steps:

- Checked missing values.
- Cleaned text columns.
- Converted date column into datetime format.
- Converted numerical columns into correct data types.
- Removed missing values.
- Checked duplicate records.

### 3. Exploratory Data Analysis (EDA)

The following analyses were performed:

### 📊 Region-wise Average Unemployment Rate
- Calculated average unemployment rate for different regions.
- Compared unemployment levels across states.

### 📈 Month-wise Unemployment Trend
- Analyzed how unemployment changed over time.

### 📉 Time-Series Analysis
- Compared unemployment trends of selected regions.

### 🏆 Top 10 Regions with Highest Average Unemployment
- Identified regions having the highest average unemployment rate.

### 🔥 Correlation Analysis
- Studied relationships between:
  - Unemployment Rate
  - Employment
  - Labour Participation Rate

- Visualized correlation using a heatmap.

### 🦠 Pre-COVID vs Post-COVID Analysis
Compared unemployment conditions:

- **Pre-COVID:** Before March 2020  
- **Post-COVID:** March 2020 onwards

---

## 📊 Key Findings

- Unemployment rates vary significantly across different regions of India.
- Employment conditions are different among states due to economic and industrial factors.
- The unemployment rate changed over time.
- COVID-19 had a noticeable impact on unemployment conditions.
- Some regions experienced considerably higher unemployment rates.
- Labour market indicators changed during the pandemic period.

---

## 📁 Repository Structure





---------------------------------------------------------------------------------------------------
# OIBSIP Data Science Task 3
# 🚗 Car Price Prediction with Machine Learning

> **An intelligent regression-based ML system that predicts the selling price of used cars using real-world vehicle data.**

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blue)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Buying or selling a used car can be challenging because the price depends on multiple factors like brand, age, mileage, fuel type, transmission, and market value.

This project builds an **end-to-end Machine Learning pipeline** that learns from historical car listings and predicts the expected selling price of a vehicle.

The system performs:

✨ Data cleaning & preprocessing  
🔍 Exploratory Data Analysis (EDA)  
⚙️ Feature engineering  
🤖 Multiple regression model training  
📊 Model performance comparison  
💡 Feature importance analysis  
🚘 Real-time sample price prediction  

---

# 🎯 Objective

Develop a Machine Learning model that can accurately estimate:

> **"How much should a used car sell for based on its features?"**

---

# 📂 Dataset

The project uses:

`car_data.xls`

A dataset containing **301 used car listings** with important vehicle information.

| Feature | Description |
|---------|-------------|
| 🚘 Car_Name | Vehicle model name |
| 📅 Year | Manufacturing year |
| 💰 Selling_Price | Final selling price (Lakhs) |
| 🏷️ Present_Price | Current showroom price (Lakhs) |
| 🛣️ Kms_Driven | Distance travelled |
| ⛽ Fuel_Type | Petrol / Diesel / CNG |
| 🤝 Seller_Type | Dealer / Individual |
| ⚙️ Transmission | Manual / Automatic |
| 👤 Owner | Previous ownership count |

---

# 🗂️ Project Structure
---------------------------------------------------------------------------------------------------

