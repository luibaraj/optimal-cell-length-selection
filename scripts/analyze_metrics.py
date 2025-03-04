import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def visualize_runtime_diff(df, title):
    """
    Creates a 3D scatter plot of formulation parameters (chiN, fA, tau)
    with the runtime difference used as the color.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(df['chiN'], df['fA'], df['tau'],
                    c=df['runtime_diff'], cmap='viridis', marker='o', alpha=0.8)
    
    ax.set_xlabel('chiN')
    ax.set_ylabel('fA')
    ax.set_zlabel('tau')
    ax.set_title(title)
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Absolute Runtime Difference (sec)')
    plt.show()


def visualize_iters_diff(df, title):
    """
    Creates a 3D scatter plot of formulation parameters (chiN, fA, tau)
    with the iteration difference used as the color.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(df['chiN'], df['fA'], df['tau'],
                    c=df['iters_diff'], cmap='viridis', marker='o', alpha=0.8)
    
    ax.set_xlabel('chiN')
    ax.set_ylabel('fA')
    ax.set_zlabel('tau')
    ax.set_title(title)
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Absolute Iteration Difference')
    plt.show()


def main():

    df = pd.read_csv('/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/experiment_rslts/metrics.csv')

    
    # pivot df by formulation parameters
    df_pivot = df.pivot_table(index=['chiN', 'fA', 'tau'],
                              columns='group',
                              values=['Total_Runtime', 'iters']).reset_index()
    df_pivot.columns = ['_'.join(col).strip('_') for col in df_pivot.columns.values]

    
    # Compute the absolute differences for runtime and iteration counts.
    df_pivot['runtime_diff'] = abs(df_pivot['Total_Runtime_predicted_L'] - df_pivot['Total_Runtime_nearest_L'])
    df_pivot['iters_diff'] = abs(df_pivot['iters_predicted_L'] - df_pivot['iters_nearest_L'])


    visualize_runtime_diff(df_pivot, "3D Scatter Plot of Runtime Difference")
    visualize_iters_diff(df_pivot, "3D Scatter Plot of Iteration Difference")


if __name__ == "__main__":
    main()
