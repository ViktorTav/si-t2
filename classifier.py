import csv
import json
from time import time

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 42

DATASET_FILE = "classification.txt"
MLP_RESULTS = "mlp_c_results.json"
RF_RESULTS = "rf_c_results.json"

TEST_SPLIT_SIZE = 0.3

EPOCHS_LIST = [100, 1000, 10000]
HIDDEN_LAYERS_LIST = [(4,2),(8,4),(16,8)]
LEARNING_RATES = [2, 1.5, 1, 0.1, 0.01, 0.001, 0.0001]

TREES_LIST = [10, 100, 250]
MAX_FEATURES_LIST = [2, 3]
CRITERION = 'gini'

def load_dataset(file, test_split_size):
    columns = ['ID', 'Sinal1', 'Sinal2', 'Sinal3', 'Sinal4', 'Sinal5', 'Gravidade', 'Label']
    df = pd.read_csv(file, names=columns)

    x = df.drop(['ID', 'Sinal1', 'Sinal2', 'Gravidade', 'Label'], axis=1)
    y = df['Label']

    return train_test_split(x, y, test_size=test_split_size, random_state=RANDOM_SEED)

def scale_dataset(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled

def random_forest(trees, max_features, criterion, x_train, x_test, y_train, y_test, return_model=False):
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

    training_time = time()-start

    print(f" Training time: {(training_time):.2f}s")
    print("----------------------------------\n")

    if return_model:
        return rf

    rf_pred = rf.predict(x_test)

    return {
        'accuracy_score': round(accuracy_score(y_test, rf_pred), 4),
        'precision_score': round(precision_score(y_test, rf_pred, average='macro', zero_division=0.0),4),
        'recall_score': round(recall_score(y_test, rf_pred, average='macro'),4),
        'f1_score': round(f1_score(y_test, rf_pred, average='macro'),4),
        'training_time': round(training_time,2)
    }

def mlp(hidden_layers, epochs, learning_rate, batch_size, x_train, x_test, y_train, y_test, return_model=False):
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
        activation='tanh',
        solver='sgd',
        learning_rate='constant',
        momentum=0.0,
        nesterovs_momentum=False,
        alpha=0.0,
        n_iter_no_change=epochs
    )

    start = time()

    mlp.fit(x_train, y_train)

    training_time = time()-start

    print(f" Training time: {(training_time):.2f}s")
    print("----------------------------------\n")

    if return_model:
        return mlp

    mlp_pred = mlp.predict(x_test)

    return {
        'accuracy_score': round(accuracy_score(y_test, mlp_pred), 4),
        'precision_score': round(precision_score(y_test, mlp_pred, average='macro', zero_division=0.0),4),
        'recall_score': round(recall_score(y_test, mlp_pred, average='macro'),4),
        'f1_score': round(f1_score(y_test, mlp_pred, average='macro'),4),
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

                if best_result is None or result['f1_score'] > best_result['f1_score']:
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

            if best_result is None or result['f1_score'] > best_result['f1_score']:
                best_result = result 

            results.append(result)

    with open(RF_RESULTS, mode="w", encoding="utf-8", newline="") as file:
        file.write(json.dumps(results, indent=4))

    return best_result

def test():
    x_train, x_test, y_train, y_test = load_dataset(DATASET_FILE, TEST_SPLIT_SIZE)
    x_train_scaled, x_test_scaled = scale_dataset(x_train, x_test)

    rf_result = test_random_forest(TREES_LIST, MAX_FEATURES_LIST, CRITERION, x_train, x_test, y_train, y_test)
    mlp_result = test_mlp(LEARNING_RATES, HIDDEN_LAYERS_LIST, EPOCHS_LIST, x_train_scaled, x_test_scaled, y_train, y_test)

    print("\n---------- Random Forest - Best Result ----------")
    print(rf_result)

    print("\n--------------- MLP - Best Result ---------------")
    print(mlp_result)

def main():
    x_train, x_test, y_train, y_test = load_dataset(DATASET_FILE, TEST_SPLIT_SIZE)
    x_train_scaled, x_test_scaled = scale_dataset(x_train, x_test)

    mlp_model = mlp((16,8), 10000, 0.1, len(x_train_scaled), x_train_scaled, x_test_scaled, y_train, y_test, True)
    rf_model = random_forest(100, 3, 'gini', x_train, x_test, y_train, y_test, True)

    mlp_pred = mlp_model.predict(x_test_scaled)
    rf_pred = rf_model.predict(x_test)

    print("--- Random Forest ---")
    print(f" Accuracy score: {accuracy_score(y_test, rf_pred):.2%}")
    print(f" Precision score: {precision_score(y_test, rf_pred, average='macro'):.2%}")
    print(f" Recall score: {recall_score(y_test, rf_pred, average='macro'):.2%}")
    print(f" F1 score: {f1_score(y_test, rf_pred, average='macro'):.2%}")

    print(classification_report(y_test, rf_pred))

    print("\n\n")

    print("--- MLP ---")
    print(f" Accuracy score: {accuracy_score(y_test, mlp_pred):.2%}")
    print(f" Precision score: {precision_score(y_test, mlp_pred, average='macro', zero_division=0.0):.2%}")
    print(f" Recall score: {recall_score(y_test, mlp_pred, average='macro'):.2%}")
    print(f" F1 score: {f1_score(y_test, mlp_pred, average='macro'):.2%}")

    print(classification_report(y_test, mlp_pred))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Matriz de Confusão: Random Forest')
    ax[0].set_xlabel('Predito')
    ax[0].set_ylabel('Real')

    sns.heatmap(confusion_matrix(y_test, mlp_pred), annot=True, fmt='d', cmap='Greens', ax=ax[1])
    ax[1].set_title('Matriz de Confusão: MLP')
    ax[1].set_xlabel('Predito')
    ax[1].set_ylabel('Real')

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()