import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# calling of dataset
df=pd.read_csv("data/data.csv")
pd.set_option('display.max_columns', None)


# Cleaning

df=df.drop("timestamp", axis=1)
objects=['equipment_energy_consumption','lighting_energy','zone1_temperature','zone1_humidity','zone2_temperature']
df[objects]=df[objects].apply(pd.to_numeric, errors='coerce')
# corr=df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()

# sns.histplot(df['outdoor_temperature'], bins=30, kde=True)
# plt.title('Distribution of Equipment Energy Consumption')   
# plt.show()



obj=['equipment_energy_consumption', 'lighting_energy', 'zone1_temperature',
       'zone1_humidity', 'zone2_temperature', 'zone2_humidity',
       'zone3_temperature', 'zone3_humidity', 'zone4_temperature',
       'zone4_humidity', 'zone5_temperature', 'zone5_humidity',
       'zone6_temperature', 'zone6_humidity', 'zone7_temperature',
       'zone7_humidity', 'zone8_temperature', 'zone8_humidity',
       'zone9_temperature', 'zone9_humidity', 'outdoor_temperature',
       'atmospheric_pressure', 'outdoor_humidity', 'wind_speed',
       'visibility_index', 'dew_point', 'random_variable1',
       'random_variable2']
print(df.columns)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[obj] = imputer.fit_transform(df[obj])
print(df.isnull().sum())


# Normalization/Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[obj] = scaler.fit_transform(df[obj])
print(df.describe())

# sns.histplot(df['equipment_energy_consumption'], bins=30, kde=True)
# plt.title('Distribution of Equipment Energy Consumption')   
# plt.show()

#Fearure Selection
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

select=SelectKBest(score_func=f_regression, k=10)
X = df.drop('equipment_energy_consumption', axis=1)
y = df['equipment_energy_consumption']
X_new = select.fit_transform(X, y)
mask = select.get_support()
selected_features = X.columns[mask]
print("Selected features:", selected_features.tolist())


