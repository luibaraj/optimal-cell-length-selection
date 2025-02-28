
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# STEP 1: Load Main Data for a Specific Phase
# -------------------------------
def load_main_data(file_path, phase):
    """
    Load the main data file and filter by the specified phase.
    """
    df = pd.read_csv(file_path, delimiter="\t", skiprows=1, names=[
        "chiN", "fA", "tau", "phase", "H", "H1", "L", "Runtime", "Iterations", "STATUS"
    ])
    return df[df["phase"] == phase]

# -------------------------------
# STEP 2: Load Good Formulations for a Phase
# -------------------------------
def load_good_formulations(formulations_file):
    """
    Load the good formulations CSV and format the values to six decimals.
    Returns a dictionary with sets for 'chiN', 'fA', and 'tau'.
    """
    formulations = pd.read_csv(formulations_file)
    good_formulations = {
        'chiN': set(formulations['chiN'].apply(lambda x: f"{x:.6f}")),
        'fA': set(formulations['fA'].apply(lambda x: f"{x:.6f}")),
        'tau': set(formulations['tau'].apply(lambda x: f"{x:.6f}"))
    }
    return good_formulations

# -------------------------------
# STEP 3: Filter Data Based on Good Formulations
# -------------------------------
def filter_data_by_formulations(df, good_formulations):
    """
    Filter rows in the dataframe where chiN, fA, and tau match the good formulations.
    """
    chiN_str = df['chiN'].apply(lambda x: f"{x:.6f}")
    fA_str = df['fA'].apply(lambda x: f"{x:.6f}")
    tau_str = df['tau'].apply(lambda x: f"{x:.6f}")
    
    mask = chiN_str.isin(good_formulations['chiN']) & \
           fA_str.isin(good_formulations['fA']) & \
           tau_str.isin(good_formulations['tau'])
    return df[mask]

# -------------------------------
# STEP 4: Preprocess Features and Target Variable
# -------------------------------
def preprocess_features(df):
    """
    Split the dataframe into features (X) and target (y).
    """
    X = df[['chiN', 'fA', 'tau']]
    y = df['L']
    return X, y

# -------------------------------
# STEP 5: Train the XGBoost Model and Return Test Set
# -------------------------------
def train_xgb_model(X, y, test_size=0.2, random_state=42):
    """
    Scale the features, split the data, train an XGBoost regressor,
    print the mean squared error, and return the trained model, scaler,
    and the test set (scaled).
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=test_size, random_state=random_state
    )
    
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    return model, scaler, X_test

# -------------------------------
# STEP 6: Predict on Test Set, Uniformly Sample n Simulations, and Save Predictions
# -------------------------------
def predict_on_test_set(model, scaler, X_test, output_file, n):
    """
    Inverse transform the test set to get original formulations, create keys,
    predict L, uniformly sample n simulations, and save predictions to CSV.
    """
    # Inverse transform to get original feature values
    X_test_orig = scaler.inverse_transform(X_test)
    keys = [f"{row[0]:.6f}_{row[1]:.6f}_{row[2]:.6f}" for row in X_test_orig]
    
    # Predict L on the scaled test set
    y_pred = model.predict(X_test)
    
    # Create DataFrame with keys and predicted L values
    df_pred = pd.DataFrame({'key': keys, 'L': y_pred})
    
    # Uniformly sample n simulations (if there are at least n)
    if len(df_pred) > n:
        df_pred_sampled = df_pred.sample(n=n, random_state=42)
    else:
        df_pred_sampled = df_pred
    
    df_pred_sampled.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# -------------------------------
# STEP 7: Main Execution Function
# -------------------------------
def main():
    # IMPORTANT PRE-REQ:
    # Split the clustered_density_data output into separate files for good and bad densities.
    # good_formulations_file should have the clusters of densities that converged properly


    data_file = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    good_formulations_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/predicted_L.csv"
    phase = 'HEX'
    
    # Load and filter the main data by phase
    df = load_main_data(data_file, phase)
    
    # Load good formulations and filter the data accordingly
    good_formulations = load_good_formulations(good_formulations_file)
    df_filtered = filter_data_by_formulations(df, good_formulations)
    
    # Preprocess features and target variable
    X, y = preprocess_features(df_filtered)
    
    # Train the XGBoost model and get the test set (scaled)
    model, scaler, X_test = train_xgb_model(X, y)
    
    # Predict on the test set and save the results
    predict_on_test_set(model, scaler, X_test, output_file, 50)


if __name__ == "__main__":
    main()