
# AI Salary Prediction System

## Overview

This project analyzes and predicts salaries in the Artificial Intelligence and Machine Learning job market using different machine learning techniques.
The system includes regression models, a classification model, and a graphical user interface (GUI) that allows users to enter job and company details and receive salary predictions.

The main goal is to determine whether job characteristics, company attributes, and work conditions can accurately predict AI/ML salaries.

---

## Dataset Description

* Source: Global AI & Machine Learning Salaries dataset
* Number of rows: 20,407
* Number of columns: 11

### Main Variables

work_year: Year of employment
experience_level: EN, MI, SE
employment_type: FT, PT, CT, FL
job_title: Job role
salary: Salary in original currency
salary_currency: Salary currency
salary_in_usd: Salary converted to USD (target variable)
employee_residence: Employee country
remote_ratio: Percentage of remote work
company_location: Company country
company_size: S, M, L

---

## Data Cleaning and Preprocessing

* Removed duplicate records
* No missing values in the dataset
* Checked salary outliers and kept extreme values
* Applied ordinal encoding to company size and employment type
* Applied one-hot encoding to job title, currency, residence, company location, and experience level
* Applied StandardScaler to numerical features
* Built a preprocessing pipeline using ColumnTransformer and Pipeline

---

## Descriptive Statistics

Mean salary (USD): 155,086
Median salary (USD): 144,000
Standard deviation: 76,384
Minimum: 15,000
Maximum: 800,000

Visualizations include: salary histogram, salary boxplot, scatter plot of remote ratio vs salary.

---

## Machine Learning Models

### 1. Multiple Linear Regression

R²: 0.1348
RMSE: 72,033
MAE: 31,533

Linear regression did not capture the complex patterns in the dataset.

### 2. Gradient Boosting Regression

Test R²: 0.9892
RMSE: 8,045
MAE: 809

This model performed extremely well and captured non-linear relationships effectively.

### 3. Logistic Regression (Classification)

Salary was divided into:
1 = High salary (above median)
0 = Low salary (median or below)

Accuracy: 89.47%
F1-score: about 0.89 for both classes

---

## GUI Application

A Tkinter-based GUI was developed to allow users to interact with the system without writing code.

### User Inputs

* Model selection: regression or classification
* Work year
* Experience level
* Employment type
* Job title
* Salary
* Salary currency
* Employee residence
* Remote ratio
* Company location
* Company size

### Output

* Predicted salary in USD (for regression)
* Salary category: High or Low (for classification)

The GUI connects directly to the trained models for real-time predictions.

---

## Suggested Project Structure

```
data/
models/
gui/
notebooks/
README.md
requirements.txt
```

---

## How to Run the Project

Install the required libraries:

```
pip install -r requirements.txt
```

Run the GUI:

```
python gui/app.py
```



## Future Improvements

* Add more advanced machine learning models
* Deploy the project as a web application
* Add a visualization dashboard
* Expand the dataset with more global data



## Conclusion

The project shows that machine learning models, especially Gradient Boosting, can effectively predict AI and ML salaries. The GUI makes the tool easy to use for anyone interested in salary prediction or analysis.
