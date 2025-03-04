import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def main():
    # modify these variables if needed
    data_file = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    good_formulations_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/exp_predicted_L.csv"
    phase = 'HEX'
    n_samples = 100  # Number of simulations to sample
    test_size = 0.2
    random_state = 42

    # -------------------------------
    # STEP 1: Load Main Data for a Specific Phase
    # -------------------------------
    column_names = [
        "chiN", "fA", "tau", "phase", "H", "H1", "L", "Runtime", "Iterations", "STATUS"
    ]
    df = pd.read_csv(data_file, delimiter="\t", skiprows=1, names=column_names)
    df = df[df["phase"] == phase]

    # -------------------------------
    # STEP 2: Load Good Formulations and Filter Data
    # -------------------------------
    formulations = pd.read_csv(good_formulations_file)
    # Format the values to six decimals and create sets for each parameter
    good_formulations = {
        'chiN': set(formulations['chiN'].apply(lambda x: f"{x:.6f}")),
        'fA': set(formulations['fA'].apply(lambda x: f"{x:.6f}")),
        'tau': set(formulations['tau'].apply(lambda x: f"{x:.6f}"))
    }
    
    # Convert the main data parameters to six-decimal string representations and filter
    chiN_str = df['chiN'].apply(lambda x: f"{x:.6f}")
    fA_str = df['fA'].apply(lambda x: f"{x:.6f}")
    tau_str = df['tau'].apply(lambda x: f"{x:.6f}")
    mask = chiN_str.isin(good_formulations['chiN']) & \
           fA_str.isin(good_formulations['fA']) & \
           tau_str.isin(good_formulations['tau'])
    df_filtered = df[mask]

    # -------------------------------
    # STEP 3: Seperate Features and Target Variable
    # -------------------------------
    X = df_filtered[['chiN', 'fA', 'tau']]
    y = df_filtered['L']

    # -------------------------------
    # STEP 4: Train the XGBoost Model
    # -------------------------------
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

    # -------------------------------
    # STEP 5: Predict on Test Set and Save Predictions
    # -------------------------------
    # Inverse transform to retrieve original feature values
    X_test_orig = scaler.inverse_transform(X_test)
    # Create unique keys for each row based on formatted feature values
    keys = [f"{row[0]:.6f}_{row[1]:.6f}_{row[2]:.6f}" for row in X_test_orig]
    
    # Predict L for the test set
    y_pred = model.predict(X_test)
    df_pred = pd.DataFrame({'key': keys, 'L': y_pred})
    
    # Uniformly sample n simulations if there are more than n rows
    if len(df_pred) > n_samples:
        df_pred_sampled = df_pred.sample(n=n_samples, random_state=random_state)
    else:
        df_pred_sampled = df_pred

    df_pred_sampled.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
