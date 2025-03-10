# Student Depression Dataset - SVM Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="whitegrid")

# prepare
main = pd.read_csv("Student Depression Dataset.csv")
main.dropna(thresh=len(main)*0.7, axis=1, inplace=True)
main.dropna(inplace=True)
pdf = PdfPages("depression svm_classification_report.pdf")

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

# Display encode
if encoding_summary:
    encoding_df = pd.concat(encoding_summary, ignore_index=True)
    print("\nFull Encoding Mapping Table:")
    print(encoding_df)

    for col in encoding_df['Column'].unique():
        print(f"\nColumn: {col}")
        print(encoding_df[encoding_df['Column'] == col])

# Depression Label Distribution
if 'Depression' in main.columns:
    fig1 = plt.figure(figsize=(6,4))
    sns.countplot(x='Depression', data=main)
    plt.title('Depression Label Distribution')
    plt.xlabel('Depression (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    pdf.savefig(fig1)
    plt.close()

# Correlation Heatmap
fig2 = plt.figure(figsize=(12,10))
sns.heatmap(main.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
pdf.savefig(fig2)
plt.close()

# training
X = main.drop(columns=['Depression'])
y = main['Depression']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 80/20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("SVM classification model completed")
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(report)

# Save text page to PDF
fig_text = plt.figure(figsize=(8.5,11))
text = f"SVM classification model completed\n\nAccuracy: {acc:.4f}\n\nClassification Report:\n\n{report}"
plt.axis('off')
plt.text(0.01, 0.99, text, va='top', ha='left', fontsize=10, wrap=True)
pdf.savefig(fig_text)
plt.close()

# Plot Confusion Matrix
fig_cm = plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()
pdf.savefig(fig_cm)
plt.close()

# Classification Report Plot
fig_cls = plt.figure(figsize=(8,5))
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'])
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=plt.gca())
plt.title('Classification Report (Precision / Recall / F1-score)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
pdf.savefig(fig_cls)
plt.close()

# Close PDF
pdf.close()
