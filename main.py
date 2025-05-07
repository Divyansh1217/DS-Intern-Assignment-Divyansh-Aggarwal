import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# calling of dataset
df=pd.read_csv("data/data.csv")
pd.set_option('display.max_columns', None)
print(df.describe())
print(df.info())

# Cleaning
df=df.drop("timestamp", axis=1)
objects=['equipment_energy_consumption','lighting_energy','zone1_temperature','zone1_humidity','zone2_temperature']
df[objects]=df[objects].apply(pd.to_numeric, errors='coerce')
corr=df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()