# Responses2 Dataset - SVR Regression and Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")
main = pd.read_csv("responses2.csv")

# prepare
# Process Social Time: extract average hours
def extract_social_time(value):
    if isinstance(value, str) and '-' in value:
        parts = value.replace('hours', '').strip().split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    elif isinstance(value, str) and value.strip().replace('.', '').isdigit():
        return float(value.strip())
    else:
        return np.nan

if 'Social Time' in main.columns:
    main['Social Time'] = main['Social Time'].apply(extract_social_time)
main.dropna(thresh=len(main)*0.7, axis=1, inplace=True)
main.dropna(inplace=True)

# Encode
encoding_summary = []
label_encoders = {}
for col in main.select_dtypes(include='object').columns:
    le = LabelEncoder()
    main[col] = le.fit_transform(main[col])
    label_encoders[col] = le
    encoding_summary.append(pd.DataFrame({
        'Column': col,
        'Original Value': le.classes_,
        'Encoded Value': list(range(len(le.classes_)))
    }))

# Display encoding table
if encoding_summary:
    encoding_df = pd.concat(encoding_summary, ignore_index=True)
    print("\nFull Encoding Mapping Table:")
    print(encoding_df)

# Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(main.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Classification
main['Stress_Level'] = main['Overall_Stress'].apply(lambda x: 0 if x <= 5 else 1)
X_cls = main.drop(columns=['Overall_Stress', 'Stress_Level'])
y_cls = main['Stress_Level']

scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)

svc_model = SVC(kernel='rbf')
svc_model.fit(X_train_cls, y_train_cls)

y_pred_cls = svc_model.predict(X_test_cls)
acc = accuracy_score(y_test_cls, y_pred_cls)
cm = confusion_matrix(y_test_cls, y_pred_cls)
report_cls = classification_report(y_test_cls, y_pred_cls, zero_division=0)

print("\n=== SVM Classification Report ===")
print(f"Accuracy: {acc:.4f}")
print(report_cls)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix (Classification)")
plt.tight_layout()
plt.show()

cls_dict = classification_report(y_test_cls, y_pred_cls, output_dict=True, zero_division=0)
cls_df = pd.DataFrame(cls_dict).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'])
cls_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8,5))
plt.title('Classification Report (Precision / Recall / F1-score)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Regression
X_reg = main.drop(columns=['Overall_Stress', 'Stress_Level'])
y_reg = main['Overall_Stress']

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_reg, y_train_reg)

y_pred_reg = svr_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== SVR Regression Report ===")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.figure(figsize=(6,5))
sns.scatterplot(x=y_test_reg, y=y_pred_reg)
plt.xlabel("True Stress Score")
plt.ylabel("Predicted Stress Score")
plt.title("SVR Prediction vs Actual")
plt.tight_layout()
plt.show()

residuals = y_test_reg - y_pred_reg
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals (Regression Errors)')
plt.xlabel('Residual (True - Predicted)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(y_test_reg.values[:50], label='True Stress Score', marker='o')
plt.plot(y_pred_reg[:50], label='Predicted Stress Score', marker='x')
plt.title('True vs Predicted Stress Score (First 50 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Stress Score')
plt.legend()
plt.tight_layout()
plt.show()
