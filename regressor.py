import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error

file_name = './regression.txt'
colunas = ['ID', 'Sinal1', 'Sinal2', 'Sinal3', 'Sinal4', 'Sinal5', 'Gravidade']
df = pd.read_csv(file_name, names=colunas)

X = df.drop(['ID', 'Sinal1', 'Sinal2'], axis=1)
y = df['Gravidade']

# Divisão em Treino (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização Z-score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Treinando Random Forest ---")
rf = RandomForestRegressor(
    n_estimators=10,
    criterion='squared_error',
    random_state=42,
    max_depth=None,
    max_features=3,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("--- Treinando Rede Neural MLP ---")
mlp = MLPRegressor(
    hidden_layer_sizes=(8,4),
    activation='logistic',
    solver='sgd',
    max_iter=10000,
    random_state=42, 
    learning_rate='constant',
    momentum=0.0,
    batch_size=len(X_train_scaled),
    nesterovs_momentum=False,
    learning_rate_init=0.01,
    alpha=0.0,
)

mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)


# 5. COMPARAÇÃO DE RESULTADOS
print("--- Random Forest ---")
print(f" Root Mean Squared Error: {root_mean_squared_error(y_test, rf_pred):.8f}")
print(f" Mean Squared Error: {mean_squared_error(y_test, rf_pred):.8f}")
print(f" Mean Absolute Error: {mean_absolute_error(y_test, rf_pred):.8f}")

# 5. COMPARAÇÃO DE RESULTADOS
print("--- MLP ---")
print(f" Root Mean Squared Error: {root_mean_squared_error(y_test, mlp_pred):.8f}")
print(f" Mean Squared Error: {mean_squared_error(y_test, mlp_pred):.8f}")
print(f" Mean Absolute Error: {mean_absolute_error(y_test, mlp_pred):.8f}")
