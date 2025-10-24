# Employee Attrition Prediction for TechNova Solutions

This repository contains the code and analysis for a project aimed at predicting employee attrition at TechNova Solutions. The goal is to build a machine learning model that can identify employees at risk of leaving, enabling the HR department to implement proactive retention strategies.

## Project Structure

- `TechNova_Attrition_Prediction_<your_id>.ipynb`: The main Jupyter Notebook containing the end-to-end implementation.
- `employee_churn_dataset.csv`: The dataset used for the analysis (this should be placed in the same directory).
- `README.md`: This file.

## Methodology

The project follows a structured data science lifecycle:

1.  **Data Understanding & EDA**: Initial analysis to understand data properties and uncover key patterns related to employee churn.
2.  **Data Preprocessing**: Cleaning the data, encoding categorical variables, and scaling numerical features.
3.  **Modeling**: Training and evaluating several classification models, including Logistic Regression, Random Forest, and Gradient Boosting.
4.  **Evaluation**: Assessing the models based on metrics suited for an imbalanced dataset (like Recall, Precision, and AUC-ROC). The significant class imbalance was identified as a major challenge.
5.  **Model Explainability**: Using SHAP (SHapley Additive exPlanations) on the best-performing baseline model (Logistic Regression) to understand the key drivers behind its predictions.
6.  **Recommendations**: Providing actionable, data-driven recommendations for the HR department based on the analysis.

## Key Findings

- **Primary Churn Drivers**: The analysis identified **low employee satisfaction**, **lower salary**, **shorter tenure**, and a **high number of overtime hours** as the most significant factors driving churn.
- **Modeling Challenge**: The dataset is highly imbalanced (approx. 80% stay vs. 20% leave). This caused the standard Random Forest and Gradient Boosting models to fail, as they predicted no churn instances at all (zero Recall).
- **Best Baseline Model**: The **Logistic Regression** model, using a `class_weight='balanced'` setting, provided a much more useful baseline. It successfully identified about half of the employees who churned, making its insights valuable despite lower overall accuracy.

## How to Run

1.  Clone this repository to your local machine.
2.  Ensure you have Python 3.x and the required libraries installed. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn shap joblib
    ```
3.  Place the `employee_churn_dataset.csv` file in the same root directory as the notebook.
4.  Open and run the `TechNova_Attrition_Prediction_<your_id>.ipynb` notebook in a Jupyter environment (like Jupyter Lab or Jupyter Notebook).
