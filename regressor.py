import json
from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

RANDOM_SEED = 42

DATASET_FILE = "regression.txt"
MLP_RESULTS = "mlp_r_results.json"
RF_RESULTS = "rf_r_results.json"

TEST_SPLIT_SIZE = 0.3

EPOCHS_LIST = [1000, 10000]
HIDDEN_LAYERS_LIST = [(4,2),(8,4),(16,8),(32,16)]
LEARNING_RATES = [1, 0.1, 0.01, 0.001]

TREES_LIST = [10, 100, 250]
MAX_FEATURES_LIST = [2, 3]
CRITERION = 'squared_error'

def load_dataset(file: str, test_split_size: float) -> list:
    columns = ['ID', 'Sinal1', 'Sinal2', 'Sinal3', 'Sinal4', 'Gravidade']
    df = pd.read_csv(file, names=columns)

    x = df.drop(['ID', 'Sinal1', 'Sinal2', 'Gravidade'], axis=1)
    y = df['Gravidade']

    return train_test_split(x, y, test_size=test_split_size, random_state=RANDOM_SEED)

def scale_dataset(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled

def random_forest(trees, max_features, criterion, x_train, x_test, y_train, y_test):
    print("\n--------- Random Forest ----------")
    print(f" Trees: {trees}")
    print(f" Max Features: {max_features}")
    print(f" Criterion: {criterion}")

    rf = RandomForestRegressor(
        n_estimators=trees,
        criterion=criterion,
        random_state=RANDOM_SEED,
        max_depth=None,
        max_features=max_features,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0
    )

    start = time()

    rf.fit(x_train, y_train)

    training_time = time()-start

    print(f" Training time: {(training_time):.2f}s")
    print("----------------------------------\n")

    rf_pred = rf.predict(x_test)

    return {
        'rmse': round(root_mean_squared_error(y_test, rf_pred), 4),
        'training_time': round(training_time,2)
    }

def mlp(hidden_layers, epochs, learning_rate, batch_size, x_train, x_test, y_train, y_test):
    print("\n-------------- MLP ---------------")
    print(f" Hidden layers: {hidden_layers}")
    print(f" Epochs: {epochs}")
    print(f" Learning rate: {learning_rate}")
    print(f" Batch size: {batch_size}")

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='logistic',
        solver='sgd',
        max_iter=epochs,
        random_state=RANDOM_SEED, 
        learning_rate='constant',
        momentum=0.0,
        batch_size=batch_size,
        nesterovs_momentum=False,
        learning_rate_init=learning_rate,
        alpha=0.0,
    )

    start = time()

    mlp.fit(x_train, y_train)

    training_time = time()-start

    print(f" Training time: {(training_time):.2f}s")
    print("----------------------------------\n")

    mlp_pred = mlp.predict(x_test)

    return {
        'rmse': round(root_mean_squared_error(y_test, mlp_pred), 4),
        'training_time': round(training_time,2)
    }

def test_mlp(learning_rates, hidden_layers_list, epochs_list, x_train_scaled, x_test_scaled, y_train, y_test):
    results = []
    best_result = None

    for hidden_layers in hidden_layers_list:
        for learning_rate in learning_rates:
            for epochs in epochs_list:
                result = mlp(hidden_layers, epochs, learning_rate, len(x_train_scaled), x_train_scaled, x_test_scaled, y_train, y_test)

                result['hidden_layers'] = hidden_layers
                result['learning_rate'] = learning_rate
                result['epochs'] = epochs

                if best_result is None or result['rmse'] < best_result['rmse']:
                    best_result = result 

                results.append(result)

    with open(MLP_RESULTS, mode="w", encoding="utf-8", newline="") as file:
        file.write(json.dumps(results, indent=4))

    return best_result

def test_random_forest(trees_list, max_features_list, criterion, x_train, x_test, y_train, y_test):
    results = []
    best_result = None
    
    for trees in trees_list:
        for max_features in max_features_list:
            result = random_forest(trees, max_features, criterion, x_train, x_test, y_train, y_test)

            result['trees'] = trees
            result['max_features'] = max_features
            result['criterion'] = criterion

            if best_result is None or result['rmse'] < best_result['rmse']:
                best_result = result 

            results.append(result)

    with open(RF_RESULTS, mode="w", encoding="utf-8", newline="") as file:
        file.write(json.dumps(results, indent=4))

    return best_result

def main():
    x_train, x_test, y_train, y_test = load_dataset(DATASET_FILE, TEST_SPLIT_SIZE)
    x_train_scaled, x_test_scaled = scale_dataset(x_train, x_test)

    rf_result = test_random_forest(TREES_LIST, MAX_FEATURES_LIST, CRITERION, x_train, x_test, y_train, y_test)
    mlp_result = test_mlp(LEARNING_RATES, HIDDEN_LAYERS_LIST, EPOCHS_LIST, x_train_scaled, x_test_scaled, y_train, y_test)

    print("\n---------- Random Forest - Best Result ----------")
    print(rf_result)

    print("\n--------------- MLP - Best Result ---------------")
    print(mlp_result)

if __name__ == "__main__":
    main()
