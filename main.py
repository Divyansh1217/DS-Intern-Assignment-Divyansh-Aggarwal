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

    imputer = KNNImputer(n_neighbors=9, weights='distance', metric='nan_euclidean')
    df[obj] = imputer.fit_transform(df[obj])
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
    df['outdoor_humidity'] = df['outdoor_humidity']**2
    from sklearn.preprocessing import power_transform
    df['outdoor_humidity'] = power_transform(df[['outdoor_humidity']], method='yeo-johnson')

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

    from sklearn.decomposition import PCA

    pca = PCA(n_components=12)
    y=df['equipment_energy_consumption']
    X=df.drop(['equipment_energy_consumption'], axis=1)
    X_new = pca.fit_transform(X)


    x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    # model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model = XGBRegressor(n_estimators=150, max_depth=15,learning_rate=0.055,subsample=0.7, random_state=42)
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




def show_histogram():
    df.hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    corr=df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

    df.plot(kind='box', figsize=(15, 8), vert=False)
    plt.title("Box-and-Whisker Plot of All Numerical Columns")
    plt.show()


if __name__ == "__main__":
    df=load()
    df=cleaning(df)

    var=df.var()
    print("Variance of each column:\n", var)
    

    df=normalization(df)
    feature_selection(df)

    show_histogram()