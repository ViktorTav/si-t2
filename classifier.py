import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

file_name = './with_label.txt'
colunas = ['ID', 'Sinal1', 'Sinal2', 'Sinal3', 'Sinal4', 'Sinal5', 'Sinal6', 'Label']
df = pd.read_csv(file_name, names=colunas)

X = df.drop(['ID', 'Sinal1', 'Sinal2', 'Label'], axis=1)
y = df['Label']

# Divisão em Treino (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização Z-score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Treinando Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=10,
    criterion='gini',
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
mlp = MLPClassifier(
    hidden_layer_sizes=(4,2),
    activation='logistic',
    solver='sgd',
    max_iter=1000,
    random_state=42, 
    learning_rate='constant',
    momentum=0.0,
    batch_size=len(X_train_scaled),
    nesterovs_momentum=False,
    learning_rate_init=1.6,
    alpha=0.0,
)

mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)


print("--- Random Forest ---")
print(f" Accurancy score: {accuracy_score(y_test, rf_pred):.2%}")
print(f" Precision score: {precision_score(y_test, rf_pred, average='macro'):.2%}")
print(f" Recall score: {recall_score(y_test, rf_pred, average='macro'):.2%}")
print(f" F1 score: {f1_score(y_test, rf_pred, average='macro'):.2%}")

print("\n\n")

print("--- MLP ---")
print(f" Accurancy score: {accuracy_score(y_test, mlp_pred):.2%}")
print(f" Precision score: {precision_score(y_test, mlp_pred, average='macro', zero_division=0.0):.2%}")
print(f" Recall score: {recall_score(y_test, mlp_pred, average='macro'):.2%}")
print(f" F1 score: {f1_score(y_test, mlp_pred, average='macro'):.2%}")


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