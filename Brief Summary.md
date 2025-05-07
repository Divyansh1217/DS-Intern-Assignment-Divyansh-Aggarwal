Energy Consumption Analysis Report

1. Problem Statement

The objective is to analyze energy consumption data and develop a machine learning model to predict equipment energy consumption. Based on the analysis and modeling, recommendations will be provided to reduce energy consumption.

2. Approach

Data Loading & Cleaning: Loaded dataset and handled missing values using both SimpleImputer (mean strategy) and KNNImputer for comparison.

Outlier Handling: Applied Interquartile Range (IQR) technique to cap outliers.

Feature Engineering:

Standardized numeric features using StandardScaler.

Applied PCA and SelectKBest for dimensionality reduction and feature selection.

Model Building:

Trained multiple models including XGBoost Regressor.

Evaluated models using Mean Squared Error (MSE), R-squared (R²), and Cross-validation scores.

3. Key Insights from the Data

Several features including outdoor_humidity showed left-skewed distributions.

Strong correlations observed between zone temperatures and equipment energy consumption.

PCA and SelectKBest both helped in reducing dimensionality; PCA gave better overall performance.

Model:
XGBoost 
R^2 : 0.30
MSE:60.0



5. Recommendations to Reduce Equipment Energy Consumption

Optimize Indoor Climate Control:

Improve HVAC scheduling based on zone-specific temperature and humidity data.

Automate adjustments using predictive models.

Enhance Insulation & Ventilation:

Better insulation in high-consumption zones.

Improve airflow where humidity and temperature are harder to regulate.

Intelligent Lighting:

Reduce lighting energy by adopting occupancy sensors and LED fixtures.

Predictive Maintenance:

Use model predictions to identify peak consumption patterns and flag anomalies.

Renewable Integration:

Consider solar-powered support for equipment to offset peak load times.

6. Conclusion

This analysis demonstrated the effectiveness of PCA and XGBoost for energy prediction. By understanding the influencing factors and their dynamics, actionable recommendations were developed to drive efficiency and reduce equipment energy consumption.