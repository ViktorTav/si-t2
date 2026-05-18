from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 42
DATASET_FILE = "classification.txt"
TEST_SPLIT_SIZE = 0.3

HIDDEN_LAYERS = (4,2)
EPOCHS = 1000
LEARNING_RATE = 1.6

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

    print("------------- Metrics ------------")
    print(f" Accurancy score: {accuracy_score(y_test, rf_pred):.2%}")
    print(f" Precision score: {precision_score(y_test, rf_pred, average='macro'):.2%}")
    print(f" Recall score: {recall_score(y_test, rf_pred, average='macro'):.2%}")
    print(f" F1 score: {f1_score(y_test, rf_pred, average='macro'):.2%}")
    print("----------------------------------")

def mlp(hidden_layers, epochs, learning_rate, batch_size, x_train, x_test, y_train, y_test):
    print("\n-------------- MLP ---------------")
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

    print("--------------Metrics-------------")
    print(f" Accurancy score: {accuracy_score(y_test, mlp_pred):.2%}")
    print(f" Precision score: {precision_score(y_test, mlp_pred, average='macro', zero_division=0.0):.2%}")
    print(f" Recall score: {recall_score(y_test, mlp_pred, average='macro'):.2%}")
    print(f" F1 score: {f1_score(y_test, mlp_pred, average='macro'):.2%}")
    print("----------------------------------")


x_train, x_test, y_train, y_test = load_dataset(DATASET_FILE, TEST_SPLIT_SIZE)
x_train_scaled, x_test_scaled = scale_dataset(x_train, x_test)

random_forest(TREES, MAX_FEATURES, CRITERION, x_train, x_test, y_train, y_test)
mlp(HIDDEN_LAYERS, EPOCHS, LEARNING_RATE, len(x_train), x_train_scaled, x_test_scaled, y_train, y_test)