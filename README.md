# nba-champ-predictor
# NBA Championship Prediction

This project utilizes machine learning techniques to predict NBA championship outcomes based on historical team performance metrics. The code is organized into several key sections, each contributing to the analysis and prediction process.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data Preparation](#data-preparation)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Prediction](#prediction)
8. [Results](#results)

## Introduction

The primary objective of this project is to predict NBA championship outcomes using machine learning models. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and predictions on new data.

## Dependencies

- Pandas: Data manipulation and analysis library.
- NumPy: Numerical computing library.
- Seaborn and Matplotlib: Data visualization libraries.
- Scikit-learn: Machine learning library for model training and evaluation.
- XGBoost: Gradient boosting library for regression.

## Data Preparation

- The project reads historical NBA data and recent data from CSV files.
- Unnecessary columns are dropped to focus on relevant features.
- Variables with a Pearson's correlation coefficient greater than 0.25 with playoff wins are selected.

## Exploratory Data Analysis (EDA)

- Visualizations, including heatmaps, illustrate the correlation between selected variables and playoff wins.
- The relationship between specific variables, such as eFG% and O_BLK, is explored.

## Model Training

- The dataset is split into training and validation sets.
- Linear Regression, Random Forest Regression, and XGBoost Regression models are trained using Scikit-learn and XGBoost libraries.

## Model Evaluation

- Mean Absolute Error is used to evaluate the performance of each model on the validation set.

## Prediction

- The trained models are applied to predict playoff wins for new data representing the latest NBA season.

## Results

- Predictions are made using Linear Regression, Random Forest Regression, and XGBoost Regression models.
- The results are stored in a DataFrame and presented, showing the predicted playoff wins for each team.

Feel free to explore the Jupyter Notebooks in the `notebooks` directory for detailed insights into each step of the project.


