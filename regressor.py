from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

RANDOM_SEED = 42
DATASET_FILE = "regression.txt"
TEST_SPLIT_SIZE = 0.3

HIDDEN_LAYERS = (8,4)
EPOCHS = 10000
LEARNING_RATE = 0.01

TREES = 10
MAX_FEATURES = 3
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

    print(f" Training time: {(time()-start):.2f}s")
    print("----------------------------------\n")

    rf_pred = rf.predict(x_test)

    print("------------- Metrics ------------")
    print(f" Root Mean Squared Error: {root_mean_squared_error(y_test, rf_pred):.8f}")
    print(f" Mean Squared Error: {mean_squared_error(y_test, rf_pred):.8f}")
    print(f" Mean Absolute Error: {mean_absolute_error(y_test, rf_pred):.8f}")
    print("----------------------------------")

def mlp(hidden_layers, epochs, learning_rate, batch_size, x_train, x_test, y_train, y_test):
    print("\n-------------- MLP ---------------")
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

    print(f" Training time: {(time()-start):.2f}s")
    print("----------------------------------\n")

    mlp_pred = mlp.predict(x_test)

    print("--------------Metrics-------------")
    print(f" Root Mean Squared Error: {root_mean_squared_error(y_test, mlp_pred):.8f}")
    print(f" Mean Squared Error: {mean_squared_error(y_test, mlp_pred):.8f}")
    print(f" Mean Absolute Error: {mean_absolute_error(y_test, mlp_pred):.8f}")
    print("----------------------------------")

x_train, x_test, y_train, y_test = load_dataset(DATASET_FILE, TEST_SPLIT_SIZE)
x_train_scaled, x_test_scaled = scale_dataset(x_train, x_test)

random_forest(TREES, MAX_FEATURES, CRITERION, x_train, x_test, y_train, y_test)
mlp(HIDDEN_LAYERS, EPOCHS, LEARNING_RATE, len(x_train), x_train_scaled, x_test_scaled, y_train, y_test)