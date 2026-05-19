import csv
from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 42
DATASET_FILE = "classification.txt"
RESULTS_CSV = "classification.csv"
TEST_SPLIT_SIZE = 0.3

EPOCHS = 1000
HIDDEN_LAYERS_LIST = [(4,2),(8,4),(16,8),(32,16)]
LEARNING_RATES = [2, 1.6, 1, 0.1, 0.01, 0.001]

TREES = 1000
MAX_FEATURES = 3
CRITERION = 'gini'

def load_dataset(file: str, test_split_size: float) -> list:
    columns = ['ID', 'Sinal1', 'Sinal2', 'Sinal3', 'Sinal4', 'Sinal5', 'Sinal6', 'Label']
    df = pd.read_csv(file, names=columns)

    x = df.drop(['ID', 'Sinal1', 'Sinal2', 'Label'], axis=1)
    y = df['Label']

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
    
    start = time()

    rf = RandomForestClassifier(
        n_estimators=trees,
        criterion=criterion,
        max_features=max_features,
        random_state=RANDOM_SEED,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0
    )

    rf.fit(x_train, y_train)

    print(f" Training time: {(time()-start):.2f}s")
    print("----------------------------------\n")

    rf_pred = rf.predict(x_test)

    return {
        'accurance_score': accuracy_score(y_test, rf_pred),
        'precision_score': precision_score(y_test, rf_pred, average='macro', zero_division=0.0),
        'recall_score': recall_score(y_test, rf_pred, average='macro'),
        'f1_score': f1_score(y_test, rf_pred, average='macro')
    }

def mlp(hidden_layers, epochs, learning_rate, batch_size, x_train, x_test, y_train, y_test):
    print("\n-------------- MLP ---------------")
    print(f" Hidden layers: {hidden_layers}")
    print(f" Epochs: {epochs}")
    print(f" Learning rate: {learning_rate}")
    print(f" Batch size: {batch_size}")

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=epochs,
        random_state=RANDOM_SEED, 
        activation='logistic',
        solver='sgd',
        learning_rate='constant',
        momentum=0.0,
        nesterovs_momentum=False,
        alpha=0.0,
    )

    start = time()

    mlp.fit(x_train, y_train)

    print(f" Training time: {(time()-start):.2f}s")
    print("----------------------------------\n")

    mlp_pred = mlp.predict(x_test)

    return {
        'accuracy_score': f"{accuracy_score(y_test, mlp_pred)}",
        'precision_score': precision_score(y_test, mlp_pred, average='macro', zero_division=0.0),
        'recall_score': recall_score(y_test, mlp_pred, average='macro'),
        'f1_score': f1_score(y_test, mlp_pred, average='macro')
    }


x_train, x_test, y_train, y_test = load_dataset(DATASET_FILE, TEST_SPLIT_SIZE)
x_train_scaled, x_test_scaled = scale_dataset(x_train, x_test)

random_forest(TREES, MAX_FEATURES, CRITERION, x_train, x_test, y_train, y_test)

results = []

for hidden_layers in HIDDEN_LAYERS_LIST:
    for learning_rate in LEARNING_RATES:
        result = mlp(hidden_layers, EPOCHS, learning_rate, len(x_train_scaled), x_train_scaled, x_test_scaled, y_train, y_test)

        result['hidden_layers'] = hidden_layers
        result['learning_rate'] = learning_rate

        results.append(result)

with open(RESULTS_CSV, mode="w", encoding="utf-8", newline="") as file:
    fieldnames = ["hidden_layers", "learning_rate", 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
    writer = csv.DictWriter(file, fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)