# ============================================================
# TASK 1: Iris Flower Classification
# CodeAlpha Data Science Internship
# Student: Gatiksha | ID: CA/DF1/101914
# ============================================================

# --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   IRIS FLOWER CLASSIFICATION — CodeAlpha Internship")
print("=" * 55)

# ── 1. LOAD DATASET ─────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\n📊 Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass Distribution:")
print(df['species'].value_counts())

# ── 2. EXPLORATORY DATA ANALYSIS (EDA) ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Iris Flower — Exploratory Data Analysis', fontsize=16, fontweight='bold')

features = iris.feature_names
colors   = ['#E74C3C', '#2ECC71', '#3498DB']

for i, feature in enumerate(features[:4]):
    ax = axes[i // 2][i % 2]
    for j, species in enumerate(iris.target_names):
        subset = df[df['species'] == species][feature]
        ax.hist(subset, alpha=0.7, label=species, color=colors[j], bins=15, edgecolor='white')
    ax.set_title(f'Distribution of {feature}', fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA plot saved as 'iris_eda.png'")

# Pair Plot
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette={'setosa': '#E74C3C',
                                          'versicolor': '#2ECC71',
                                          'virginica': '#3498DB'})
plt.suptitle('Iris Pair Plot', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('iris_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Pair plot saved as 'iris_pairplot.png'")

# ── 3. FEATURE & TARGET SPLIT ───────────────────────────────
X = df[iris.feature_names]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📌 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 4. FEATURE SCALING ──────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. MODEL TRAINING ───────────────────────────────────────
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'predictions': y_pred, 'accuracy': acc}
    print(f"\n🤖 {name} Accuracy: {acc * 100:.2f}%")

# ── 6. BEST MODEL EVALUATION ────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]
print(f"\n🏆 Best Model: {best_name} ({best_model['accuracy']*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, best_model['predictions'],
                             target_names=iris.target_names))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_model['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('iris_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved as 'iris_confusion_matrix.png'")

# ── 7. FEATURE IMPORTANCE (Random Forest) ───────────────────
rf_model   = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_df    = pd.DataFrame({'Feature': iris.feature_names,
                           'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(feat_df['Feature'], feat_df['Importance'],
               color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'], edgecolor='white')
plt.title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Importance Score')
for bar, imp in zip(bars, feat_df['Importance']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{imp:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance plot saved as 'iris_feature_importance.png'")

# ── 8. SAMPLE PREDICTION ────────────────────────────────────
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
prediction = rf_model.predict(sample_scaled)
print(f"\n🌸 Sample Prediction:")
print(f"   Input: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"   Predicted Species: {iris.target_names[prediction[0]].upper()}")

print("\n" + "=" * 55)
print("   Task 1 Complete! ✅")
print("   Outputs: iris_eda.png, iris_pairplot.png,")
print("            iris_confusion_matrix.png,")
print("            iris_feature_importance.png")
print("=" * 55)
