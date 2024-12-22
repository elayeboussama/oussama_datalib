import numpy as np
import pandas as pd
import seaborn as sns
from src.datalib.visualization import plot_bar, plot_histogram, plot_correlation_matrix, plot_scatter
from src.datalib.analysis import linear_regression, polynomial_regression, multiple_linear_regression
from src.datalib.data_processing import load_csv, normalize_column, fill_missing_values, encode_categorical, scale_data
from src.datalib.statistics import calculate_mean, calculate_median, calculate_mode, perform_t_test, perform_chi_square_test
from src.datalib.supervised import train_knn, train_decision_tree, train_random_forest
from src.datalib.unsupervised import perform_kmeans, perform_pca, perform_gaussian_mixture
from src.datalib.reinforcement import basic_q_learning, sarsa

def main():
    print("Starting full datalib execution...\n")

    # Load penguins dataset
    sample_dataframe = sns.load_dataset('penguins')
    print("Sample DataFrame:\n", sample_dataframe.head(), "\n")

    # Data Processing
    print("1. Data Processing:")
    # Fill missing values for numeric columns
    filled_df = sample_dataframe.copy()
    numeric_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    for col in numeric_columns:
        filled_df = fill_missing_values(filled_df, col, method='mean')
    
    # Encode categorical variables
    categorical_columns = ['species', 'island', 'sex']
    encoded_df = filled_df.copy()
    for col in categorical_columns:
        encoded_df = encode_categorical(encoded_df, col)
    
    # Scale numeric features
    scaled_df = scale_data(encoded_df, numeric_columns)
    print("Processed DataFrame:\n", scaled_df.head())

    # Analysis
    print("\n2. Analysis:")
    # Prepare data for regression (predicting body mass from bill length)
    X = scaled_df[['bill_length_mm']].values
    y = scaled_df[['body_mass_g']].values

    model, metrics = linear_regression(X, y, return_metrics=True)
    print("Linear Regression Model (Bill Length vs Body Mass):", model)
    print("Metrics:", metrics)

    poly_model, poly_metrics = polynomial_regression(X, y, degree=2, return_metrics=True)
    print("Polynomial Regression Model:", poly_model)
    print("Metrics:", poly_metrics)

    # Multiple regression using all numeric features
    X_multi = scaled_df[numeric_columns[:-1]].values  # All numeric features except body_mass
    y_multi = scaled_df[['body_mass_g']].values
    multi_model, multi_metrics = multiple_linear_regression(X_multi, y_multi, return_metrics=True)
    print("Multiple Linear Regression Model:", multi_model)
    print("Metrics:", multi_metrics)

    # Visualization
    print("\n3. Visualization:")
    plot_bar(sample_dataframe, x='species', y='body_mass_g')
    plot_histogram(sample_dataframe, column='body_mass_g')
    plot_correlation_matrix(sample_dataframe[numeric_columns])
    plot_scatter(sample_dataframe, x='bill_length_mm', y='body_mass_g')

    # Statistics
    print("\n4. Statistics:")
    for col in numeric_columns:
        print(f"\nStatistics for {col}:")
        mean_value = calculate_mean(sample_dataframe[col])
        median_value = calculate_median(sample_dataframe[col])
        mode_value = calculate_mode(sample_dataframe[col])
        print(f"Mean: {mean_value:.2f}")
        print(f"Median: {median_value:.2f}")
        print(f"Mode: {mode_value:.2f}")

    # Supervised Learning
    print("\n5. Supervised Learning:")
    # Predict species using numeric features
    X_clf = scaled_df[numeric_columns].values
    y_clf = encoded_df['species'].values

    knn_model = train_knn(X_clf, y_clf)
    print("KNN Model (Species Classification):", knn_model)

    dt_model = train_decision_tree(X_clf, y_clf)
    print("Decision Tree Model:", dt_model)

    rf_model = train_random_forest(X_clf, y_clf)
    print("Random Forest Model:", rf_model)

    # Unsupervised Learning
    print("\n6. Unsupervised Learning:")
    # Use numeric features for clustering
    kmeans_model = perform_kmeans(X_clf, n_clusters=3)  # 3 species
    print("KMeans Model:", kmeans_model)

    pca_components, pca_variance, pca_model = perform_pca(X_clf, n_components=2)
    print("PCA Components:", pca_components)
    print("Explained Variance:", pca_variance)

    gmm_model = perform_gaussian_mixture(X_clf, n_components=3)
    print("Gaussian Mixture Model:", gmm_model)

    # Reinforcement Learning (using a simplified penguin environment)
    print("\n7. Reinforcement Learning:")
    class PenguinEnvironment:
        def __init__(self, data):
            self.data = data
            self.current_idx = 0
            # Create species to index mapping
            self.species_map = {species: idx for idx, species in 
                              enumerate(data['species'].unique())}
            self.observation_space = type('Space', (), {'n': len(self.species_map)})
            self.action_space = type('Space', (), {
                'n': 2,  # Binary action: 0 = small penguin, 1 = large penguin
                'sample': lambda: np.random.choice(2)
            })

        def reset(self):
            self.current_idx = 0
            return self._get_state()

        def _get_state(self):
            # Convert species to numeric state (0, 1, or 2)
            species = self.data['species'].iloc[self.current_idx]
            return self.species_map[species]  # Return numeric index instead of species name

        def step(self, action):
            # Get current penguin's body mass
            current_mass = self.data['body_mass_g'].iloc[self.current_idx]
            
            # Define reward logic
            # Action 0 = predict small penguin (<4000g)
            # Action 1 = predict large penguin (>=4000g)
            correct_prediction = (action == 0 and current_mass < 4000) or \
                               (action == 1 and current_mass >= 4000)
            
            reward = 1 if correct_prediction else -1
            
            # Move to next penguin
            self.current_idx = (self.current_idx + 1) % len(self.data)
            done = self.current_idx == 0
            
            return self._get_state(), reward, done, {}

    # Create environment with the penguins dataset
    env = PenguinEnvironment(sample_dataframe)

    # Run Q-learning
    q_table = basic_q_learning(env, episodes=100, alpha=0.1, gamma=0.9)
    print("\nQ-table from Q-learning:")
    print("States (species) vs Actions (0=small, 1=large):")
    print(pd.DataFrame(q_table, columns=['Predict Small', 'Predict Large']))

    # Run SARSA
    q_table_sarsa = sarsa(env, episodes=100, alpha=0.1, gamma=0.9)
    print("\nQ-table from SARSA:")
    print("States (species) vs Actions (0=small, 1=large):")
    print(pd.DataFrame(q_table_sarsa, columns=['Predict Small', 'Predict Large']))

if __name__ == "__main__":
    main()
