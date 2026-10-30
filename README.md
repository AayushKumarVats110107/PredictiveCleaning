# PredictiveCleaning
# ============================================================
# BINARY CLASSIFICATION MODEL EVALUATION
# Confusion Matrix, ROC Curve, AUC and Threshold Selection
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ============================================================
# 1. CREATE DATASET
# ============================================================

np.random.seed(42)

n = 500

# Feature 1: Cell size
cell_size = np.random.normal(50, 10, n)

# Feature 2: Cell texture
cell_texture = np.random.normal(50, 12, n)

# Create binary target
# 0 = Uninfected
# 1 = Parasitized

score = (
    0.08 * cell_size +
    0.10 * cell_texture +
    np.random.normal(0, 2, n)
)

target = (score > np.median(score)).astype(int)

data = pd.DataFrame({
    "Cell_Size": cell_size,
    "Cell_Texture": cell_texture,
    "Infected": target
})

print("\n========== DATASET ==========\n")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nClass Distribution:")
print(data["Infected"].value_counts())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = data[["Cell_Size", "Cell_Texture"]]

y = data["Infected"]


# ============================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression()

model.fit(X_train, y_train)


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

# Probability of class 1
y_probability = model.predict_proba(X_test)[:, 1]

# Default threshold = 0.5
y_prediction = (y_probability >= 0.5).astype(int)


# ============================================================
# 7. SIGMOID / PROBABILITY EXAMPLE
# ============================================================

print("\n========== PREDICTED PROBABILITIES ==========\n")

for i in range(10):
    print(
        "Actual:",
        y_test.iloc[i],
        "| Probability:",
        round(y_probability[i], 3),
        "| Prediction:",
        y_prediction[i]
    )


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_prediction
)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)


# ============================================================
# 9. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction
)

recall = recall_score(
    y_test,
    y_prediction
)

f1 = f1_score(
    y_test,
    y_prediction
)

print("\n========== MODEL PERFORMANCE ==========\n")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# ============================================================
# 10. DISPLAY CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Uninfected", "Parasitized"]
)

display.plot()

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# 11. ROC CURVE
# ============================================================

false_positive_rate, true_positive_rate, thresholds = roc_curve(
    y_test,
    y_probability
)


# ============================================================
# 12. CALCULATE AUC
# ============================================================

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== ROC AND AUC ==========\n")

print("AUC Score:", round(auc_score, 4))


# ============================================================
# 13. PLOT ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    false_positive_rate,
    true_positive_rate,
    label="ROC Curve (AUC = {:.3f})".format(auc_score)
)

# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 14. THRESHOLD SELECTION
# ============================================================

print("\n========== THRESHOLD COMPARISON ==========\n")

thresholds_to_test = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8
]

for threshold in thresholds_to_test:

    prediction = (
        y_probability >= threshold
    ).astype(int)

    accuracy_value = accuracy_score(
        y_test,
        prediction
    )

    precision_value = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall_value = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    print(
        "Threshold:",
        threshold,
        "| Accuracy:",
        round(accuracy_value, 3),
        "| Precision:",
        round(precision_value, 3),
        "| Recall:",
        round(recall_value, 3)
    )


# ============================================================
# 15. FIND BEST THRESHOLD USING YOUDEN'S J STATISTIC
# ============================================================

youden_index = true_positive_rate - false_positive_rate

best_index = np.argmax(youden_index)

best_threshold = thresholds[best_index]

print("\n========== OPTIMAL THRESHOLD ==========\n")

print(
    "Optimal Threshold:",
    round(best_threshold, 3)
)


# ============================================================
# 16. PREDICTIONS USING OPTIMAL THRESHOLD
# ============================================================

optimal_prediction = (
    y_probability >= best_threshold
).astype(int)

optimal_cm = confusion_matrix(
    y_test,
    optimal_prediction
)

print("\nConfusion Matrix Using Optimal Threshold:")
print(optimal_cm)


# ============================================================
# 17. FINAL RESULTS
# ============================================================

print("\n==========================================")
print(" BINARY CLASSIFICATION PROJECT COMPLETED")
print("==========================================")

print("\nAUC Score:",
      round(auc_score, 4))

print("Default Threshold: 0.5")

print(
    "Optimal Threshold:",
    round(best_threshold, 3)
)
