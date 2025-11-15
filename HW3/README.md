# Customer Churn Survival Analysis

This project analyzes customer churn using survival analysis and estimates customer lifetime value (CLV) to support retention strategies. Using historical customer data, it identifies at-risk customers within a year and calculates a recommended retention budget.

## Features

- **Data Preparation:** Clean and encode customer data for survival analysis.  
- **Exploratory Analysis:** Visualize distributions of tenure, age, income, churn rates, and correlations between features.  
- **Survival Modeling:** Fit multiple survival and regression models (AFT, CoxPH, Kaplan-Meier, etc.) to predict churn probabilities.  
- **Model Comparison:** Compare models using AIC, BIC, and concordance index to select the best-fitting model.  
- **CLV Calculation:** Estimate per-customer lifetime value using survival probabilities and revenue assumptions.  
- **Segment Analysis:** Analyze CLV by age group, income quartile, and customer category to find valuable customer segments.  
- **Retention Budget:** Identify at-risk customers and recommend a retention budget based on their CLV and 1-year survival probability.
