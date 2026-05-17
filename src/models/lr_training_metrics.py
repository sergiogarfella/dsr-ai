from sklearn.linear_model import LogisticRegression
import pickle

with open('data/processed/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('data/processed/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('data/processed/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('data/processed/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

logreg = LogisticRegression(penalty="l2", C=0.01, max_iter=1000, solver="saga")

logreg.fit(X_train, y_train) # Entrenamiento

# Guardar modelo LR en el paquete dsr
with open('dsr/lr_model.pkl', 'wb') as f:
    pickle.dump(logreg, f)
print("Modelo LR guardado en dsr/lr_model.pkl")




from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# Predecir
y_pred = logreg.predict(X_test)

# Get probabilities for ROC curve and AUC score
y_proba = logreg.predict_proba(X_test)[:, 1] # Probability of the positive class

# Métricas en una sola línea
print(classification_report(y_test, y_pred, target_names=['negativo', 'positivo']))

# Accuracy por separado si lo necesitas
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# AUC-ROC Score
auc_roc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC Score: {auc_roc:.4f}")

# Matriz de confusión visual
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negativo', 'positivo'])
disp.plot(cmap='Oranges')
plt.title('Matriz de Confusión')
plt.show()

# Curva ROC visual
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
