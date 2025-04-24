import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # modify these variables if needed
    data_file = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    cluster_df = pd.read_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/clustered_density_data.csv")
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/exp_predicted_L.csv"
    phase = 'GYR'
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
    formulations = cluster_df[cluster_df['dot_cluster'].isin([2, 4])]
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
    # # Inverse transform to retrieve original feature values
    # X_test_orig = scaler.inverse_transform(X_test)

    # Process bad formulations for predicting their "optimal" cell length
    bad_sims = pd.read_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/exp_nearest_L.csv")

    bad_sims[['chiN', 'fA', 'tau']] = bad_sims['key'].str.split('_', expand=True)

    bad_sims['chiN'] = bad_sims['chiN'].astype(float)
    bad_sims['fA'] = bad_sims['fA'].astype(float)
    bad_sims['tau'] = bad_sims['tau'].astype(float)

    # # Create unique keys for each row based on formatted feature values
    # keys = [f"{row[0]:.6f}_{row[1]:.6f}_{row[2]:.6f}" for row in bad_sims]
    
    # Predict L for the test set
    y_pred = model.predict(bad_sims[['chiN', 'fA', 'tau']])
    df_pred = pd.DataFrame({'key': bad_sims['key'], 'L': y_pred})
    
    # # Uniformly sample n simulations if there are more than n rows
    # if len(df_pred) > n_samples:
    #     df_pred_sampled = df_pred.sample(n=n_samples, random_state=random_state)
    # else:
    #     df_pred_sampled = df_pred

    df_pred.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # -------------------------------
    # STEP 6: Visualize the differences
    # -------------------------------


    # Load CSV files
    predicted_L_df = pd.read_csv(output_file)

    # Function to split key into x, y, z coordinates
    def split_key(df):
        coords = df['key'].str.split('_', expand=True).astype(float)
        coords.columns = ['x', 'y', 'z']
        return pd.concat([df['key'], coords, df['L']], axis=1)

    # Split and rename columns
    bad_sims = split_key(bad_sims).rename(columns={'L': 'L_nearest'})
    predicted_L_df = split_key(predicted_L_df).rename(columns={'L': 'L_predicted'})

    # Merge both dataframes on key
    merged_df = pd.merge(bad_sims, predicted_L_df, on='key')

    # Calculate the difference between L_predicted and L_nearest
    merged_df['L_diff'] = merged_df['L_predicted'] - merged_df['L_nearest']

    # Extract coordinates and L_diff for plotting
    x = merged_df['x_x']
    y = merged_df['y_x']
    z = merged_df['z_x']
    l_diff = merged_df['L_diff']

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=l_diff, cmap='coolwarm', s=50)
    ax.set_title('3D Scatter Plot of L Difference (Predicted - Nearest)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(scatter, ax=ax, label='L Difference')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
