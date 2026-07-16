"""
SIP Random Forest Baseline Model
Description: Implements Random Forest model to forecast Hourly GHI
Input: PVGIS data
Output: Forecasted GHI values
Creation Date: 06/30/2026
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def extract_data():
    """
    Extracts relevant data from the formatted JSON files and returns them as lists.
    Returns:
        Gi: Global irradiance on the inclined plane.
        Gbi: Beam (direct) irradiance on the inclined plane.
        Gdi: Diffuse irradiance on the inclined plane.
        Gri: Reflected irradiance on the inclined plane.
        H_sun: Sun height.
        T2m: 2-m air temperature.
        WS10m: 10-m total wind speed.
        hour_sin: Hour of the day with sine transformation.
        hour_cos: Hour of the day with cosine transformation.
        day_sin: Day of the year with sine transformation.
        day_cos: Day of the year with cosine transformation.
    """
    with open("../data/data1.json") as d1:        # Opens the formated JSON file without radiation components
        data1 = json.load(d1)       # Stores data in data1 variable

    with open("../data/data2.json") as d2:        # Opens the formated JSON file with radiation components
        data2 = json.load(d2)       # Stores data in data2 variable

    Gi = []     # Global irradiance on the inclined plane

    # Radiation components from data2
    Gbi = []    # Beam (direct) irradiance on the inclined plane
    Gdi = []    # Diffuse irradiance on the inclined plane
    Gri = []    # Reflected irradiance on the inclined plane

    # Meteorological variables
    H_sun = []   # Sun height
    T2m = []     # 2-m air temperature
    WS10m = []    # 10-m total wind speed
    hour_sin = []     # Hour of the day with sine transformation
    hour_cos = []     # Hour of the day with cosine transformation
    day_sin = []      # Day of the year with sine transformation
    day_cos = []      # Day of the year with cosine transformation

    for i in range(len(data1["outputs"]["hourly"])):        # Iterates through the hourly data in data1 and data2 to extract relevant variables
        Gi.append(data1["outputs"]["hourly"][i]["G(i)"])        # Adds G(i) values to the Gi list

        Gbi.append(data2["outputs"]["hourly"][i]["Gb(i)"])      # Adds Gb(i), Gd(i), and Gr(i) values to their lists
        Gdi.append(data2["outputs"]["hourly"][i]["Gd(i)"])
        Gri.append(data2["outputs"]["hourly"][i]["Gr(i)"])
        
        H_sun.append(data1["outputs"]["hourly"][i]["H_sun"])        # Adds H_sun, T2m, and WS10m values to their lists
        T2m.append(data1["outputs"]["hourly"][i]["T2m"])
        WS10m.append(data1["outputs"]["hourly"][i]["WS10m"])

        hour = int(data1["outputs"]["hourly"][i]["time"][9:11])     # Gets the hour from the time string in the data
        hour_sin.append(np.sin(2 * np.pi * hour / 24))      # Convert hour to a cyclic feature using sine transformation
        hour_cos.append(np.cos(2 * np.pi * hour / 24))      # Convert hour to a cyclic feature using cosine transformation

        dt = data1["outputs"]["hourly"][i]["time"][0:8]     # Gets the date from the time string in the data
        dt = datetime.strptime(dt, "%Y%m%d")        # Converts date string to datetime object
        day_of_year = dt.timetuple().tm_yday        # Converts date to day of the year (1-365)
        day_sin.append(np.sin(2 * np.pi * day_of_year / 365))       # Convert day of the year to a cyclic feature using sine transformation
        day_cos.append(np.cos(2 * np.pi * day_of_year / 365))       # Convert day of the year to a cyclic feature using cosine transformation

    return Gi, Gbi, Gdi, Gri, H_sun, T2m, WS10m, hour_sin, hour_cos, day_sin, day_cos       # Returns the extracted data as lists

def prepare_data(Gi, Gbi, Gdi, Gri, H_sun, T2m, WS10m, hour_sin, hour_cos, day_sin, day_cos):
    """
    Constructs the feature matrix and target vector for hourly solar irradiance forecasting.
    Parameters:
        Gi: Global irradiance on the inclined plane.
        Gbi: Beam (direct) irradiance on the inclined plane.
        Gdi: Diffuse irradiance on the inclined plane.
        Gri: Reflected irradiance on the inclined plane.
        H_sun: Sun height.
        T2m: 2-m air temperature.
        WS10m: 10-m total wind speed.
        hour_sin: Hour of the day with sine transformation.
        hour_cos: Hour of the day with cosine transformation.
        day_sin: Day of the year with sine transformation.
        day_cos: Day of the year with cosine transformation.
    Returns: 
        X: Feature matrix containing previous hourly global irradiance values. Can include radiation components and meterogical features.
        Y: Target vector containing future hourly global irradiance values.
    """
    X = []      # Input feature matrix for the model
    y = []      # Target vector for the model

    radiation_components = "Y"      # Set to "Y" to include radiation components in the model, "N" to exclude them
    meteorological_variables = "Y"      # Set to "Y" to include meteorological variables in the model, "N" to exclude them
    prev_hour = 6      # Number of previous hours to include in the model, includes current hour
    future_hour = 48     # Number of hours ahead to predict
    
    if radiation_components == "N" and meteorological_variables == "N":     # Does not include radiation components or meteorological variables in the feature matrix
        print("No radiation components. No meteorological variables included.")
        for t in range(prev_hour - 1, len(Gi) - future_hour):       # Start once enough previous hours are available and stop before future_hour samples are out of index
            X.append([Gi[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gi values
            y.append(Gi[t + future_hour])       # Target: predict the Gi future_hour hours ahead

    elif radiation_components == "Y" and meteorological_variables == "N":       # Includes radiation components, does not include meteorological variables in the feature matrix
        print("Radiation components included. No meteorological variables included.")
        for t in range(prev_hour - 1, len(Gi) - future_hour):       # Start once enough previous hours are available and stop before future_hour samples are out of index
            features = []       # Feature vector for one sample
            features.extend([Gi[t - hour] for hour in range(prev_hour)])        # Adds current and previous Gi values
            features.extend([Gbi[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gbi values
            features.extend([Gdi[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gdi values
            features.extend([Gri[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gri values
            
            X.append(features)      # Adds the feature sample to the feature matrix
            y.append(Gi[t + future_hour])       # Target: predict the Gi future_hour hours ahead

    elif radiation_components == "N" and meteorological_variables == "Y":       # Does not include radiation components, includes meteorological variables in the feature matrix
        print("No radiation components. Meteorological variables included.")
        for t in range(prev_hour - 1, len(Gi) - future_hour):       # Start once enough previous hours are available and stop before future_hour samples are out of index
            features = []       # Feature vector for one sample
            features.extend([Gi[t - hour] for hour in range(prev_hour)])        # Adds current and previous Gi values
            features.extend([
                H_sun[t],
                T2m[t],
                WS10m[t],
                hour_sin[t],        # Adds current meteorological variables
                hour_cos[t],
                day_sin[t],
                day_cos[t]
            ])

            X.append(features)      # Adds the feature sample to the feature matrix
            y.append(Gi[t + future_hour])       # Target: predict the Gi future_hour hours ahead

    elif radiation_components == "Y" and meteorological_variables == "Y":       # Includes radiation components and meteorological variables in the feature matrix
        print("Radiation components included. Meteorological variables included.")
        for t in range(prev_hour - 1, len(Gi) - future_hour):       # Start once enough previous hours are available and stop before future_hour samples are out of index
            features = []       # Feature vector for one sample
            features.extend([Gi[t - hour] for hour in range(prev_hour)])        # Adds current and previous Gi values
            features.extend([Gbi[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gbi values
            features.extend([Gdi[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gdi values
            features.extend([Gri[t - hour] for hour in range(prev_hour)])       # Adds current and previous Gri values
            features.extend([
                H_sun[t],
                T2m[t],
                WS10m[t],
                hour_sin[t],        # Adds current meteorological variables
                hour_cos[t],
                day_sin[t],
                day_cos[t]
            ])

            X.append(features)      # Adds the feature sample to the feature matrix
            y.append(Gi[t + future_hour])       # Target: predict the Gi future_hour hours ahead

    X = np.array(X)     # Casts X and y to NumPy arrays
    y = np.array(y)

    return X, y     # Returns feature matrix and target vector

def split_data(X, y):
    """
    Splits the feature matrix and target vector into training, validation, and test sets.
    Parameters:
        X: Feature matrix containing previous hourly global irradiance values. May include radiation components and meteorological features.
        y: Target vector containing future hourly global irradiance values.
    Returns:
        X_train: Feature matrix training set
        X_val: Feature matrix validation set
        X_test: Feature matrix test set
        y_train: Target vector training set
        y_val: Target vector validation set
        y_test: Target vector test set
    """
    train_end = int(len(X) * 0.70)      # End of the training set is 70% of X
    val_end = int(len(X) * 0.85)        # End of validation set is 15% of X after training set

    # Split X and y into training, validation, and test sets
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]       
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test       # Returns the training, validation, and test sets

def rndfor_train_model(X_train, y_train):
    """
    Uses X_train and y_train to train a Random Forest Regressor model.
    Parameters:
        X_train: Feature matrix training set
        y_train: Target vector training set
    Returns:
        rnd_reg: Random Forest Regressor model trained on X_train and y_train
    """
    rnd_reg = RandomForestRegressor(n_estimators=100, max_leaf_nodes=16, n_jobs=-1, random_state=42)        # Initialize Random Forest Regressor
    rnd_reg.fit(X_train, y_train)       # Fit the model to the training data
    return rnd_reg      # Returns the trained RandomForestRegressor

def predict(rnd_reg, X_test):
    """
    Uses rnd_red to make predictions on X_test.
    Parameters:
        rnd_reg: Trained RandomForestRegressor
        X_test: Feature matrix test set
    Returns:
        y_pred: Target vector predictions made by rnd_reg
    """
    y_pred = rnd_reg.predict(X_test)        # Make predictions on the test set
    return y_pred       # Returns the predictions made by rnd_reg

def error_eval(y_test, y_pred):
    """
    Calculates and displays regression evaluation metrics by comparing the predicted and actual target values.
    Parameters:
        y_test: Actual target vector from test set
        y_pred: Predicted target vector made by the RandomForestRegressor model
    Returns:
        None
    """
    # Calculates evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Displays evaluation metrics
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)

def feature_importance(rnd_reg):
    feature_names = [
        "Gi_t", "Gi_t-1", "Gi_t-2", "Gi_t-3", "Gi_t-4", "Gi_t-5", "Gi_t-6",

        "Gbi_t-1", "Gbi_t-2", "Gbi_t-3", "Gbi_t-4", "Gbi_t-5",

        "Gdi_t-1", "Gdi_t-2", "Gdi_t-3", "Gdi_t-4", "Gdi_t-5", "Gdi_t-6",

        "Gri_t-1", "Gri_t-2", "Gri_t-3", "Gri_t-4", "Gri_t-5", "Gri_t-6",

        "H_sun",
        "T2m",
        "WS10m",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos"
    ]

    for name, importance in sorted(
            zip(feature_names, rnd_reg.feature_importances_),
            key=lambda x: x[1],
            reverse=True):
        print(f"{name:10} {importance:.3f}")
    print(sum(rnd_reg.feature_importances_))

def main():
    Gi, Gbi, Gdi, Gri, H_sun, T2m, WS10m, hour_sin, hour_cos, day_sin, day_cos = extract_data()     # Extracts data from JSON files

    X, y = prepare_data(Gi, Gbi, Gdi, Gri, H_sun, T2m, WS10m, hour_sin, hour_cos, day_sin, day_cos)     # Creates feature matrix and target vector

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)       # Splits feature matrix and target vector into training, validation, and test sets

    rnd_reg = rndfor_train_model(X_train, y_train)      # Trains RandomForestRegressor model

    y_pred = predict(rnd_reg, X_test)       # Uses the trained model to make predictions on the test set

    error_eval(y_test, y_pred)      # Uses evaluation metrics to compare y_test (actual) and y_pred (predicted) values

    plt.plot(y_pred[:192], label='Predicted G(i)', color='green')       # Plots the first 192 values of y_pred
    plt.plot(y_test[:192], label='Actual G(i)', color='blue')       # Plots the first 192 values of y_test
    plt.xlabel('Time')      # Labels x-axis as Time
    plt.ylabel('G(i)')      # Labels y-axis as G(i) values
    plt.title('Hourly G(i) from 09/13/23 to 09/21/23')      # Adds title to plot
    plt.legend()        # Creates legend
    plt.grid()      # Adds grid
    plt.savefig("../plots/gi_noRad_noMet.png")      # Saves plot to plots directory
    
    #feature_importance(rnd_reg)
    #print(np.max(y))        # 1002.42
    
main()