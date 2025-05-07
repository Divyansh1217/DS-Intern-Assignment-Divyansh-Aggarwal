import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# calling of dataset
def load():
    df=pd.read_csv("data/data.csv")
    pd.set_option('display.max_columns', None)
    return df


# Cleaning
def cleaning(df):
    
    df=df.drop("timestamp", axis=1)
    objects=['equipment_energy_consumption','lighting_energy','zone1_temperature','zone1_humidity','zone2_temperature']
    df[objects]=df[objects].apply(pd.to_numeric, errors='coerce')


    global obj
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
    # from sklearn.impute import SimpleImputer
    # imputer = SimpleImputer(strategy='mean')
    # df[obj] = imputer.fit_transform(df[obj])
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)
    df[obj] = imputer.fit_transform(df[obj])
    df=df.dropna()
    def cap_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound,
                            np.where(df[column] > upper_bound, upper_bound, df[column]))


    for col in obj:
        cap_outliers_iqr(df, col)
    return df

#Normalization/Standardization
def normalization(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[obj] = scaler.fit_transform(df[obj])
    return df

#Fearure Selection
def feature_selection(df):
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score

    select=SelectKBest(score_func=f_regression, k=8)
    X = df.drop('equipment_energy_consumption', axis=1)
    y = df['equipment_energy_consumption']
    X_new = select.fit_transform(X, y)
    mask = select.get_support()
    selected_features = X.columns[mask]


    x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    # model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=10,learning_rate=0.12, random_state=42)
    # model = Lasso(alpha=0.1, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cross_val = cross_val_score(model, x_train, y_train, cv=5)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("Cross-validation scores:", cross_val)
    print("Mean cross-validation score:", cross_val.mean())








if __name__ == "__main__":
    df=load()
    df=cleaning(df)
    # sns.histplot(df['equipment_energy_consumption'], bins=30, kde=True)
    # plt.title('Distribution of Equipment Energy Consumption')   
    # plt.show()
    var=df.var()
    print("Variance of each column:\n", var)
    corr=df.corr()
    cov=df.cov()
    # print("Correlation Matrix:\n", corr)
    # print("Covariance Matrix:\n", cov)
    # sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Heatmap')
    # plt.show()

    # sns.histplot(df['outdoor_temperature'], bins=30, kde=True)
    # plt.title('Distribution of Equipment Energy Consumption')   
    # plt.show()
    df=normalization(df)
    feature_selection(df)