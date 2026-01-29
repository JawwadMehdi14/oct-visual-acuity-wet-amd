import matplotlib.pyplot as plt
import numpy as np

# Replace these with your actual data or read from CSV
years = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

acc_mean = np.array([0.6526, 0.6556, 0.6920, 0.6912, 0.6521, 0.7527, 0.8100, 0.8356, 0.7222])
acc_std = np.array([0.1367, 0.0444, 0.0726, 0.0729, 0.0713, 0.1023, 0.0400, 0.0056, 0.0667])

f1_mean = np.array([0.7718, 0.7697, 0.8037, 0.8159, 0.7711, 0.8489, 0.8917, 0.9086, 0.8295])
f1_std = np.array([0.1249, 0.0357, 0.0339, 0.0507, 0.1219, 0.0707, 0.0202, 0.00593, 0.0366])

auc_mean = np.array([0.5613, 0.5743, 0.5831, 0.5859, 0.6585, 0.7370, 0.7939, 0.5866, 0.7151])
auc_std = np.array([0.0485, 0.1353, 0.2304, 0.0767, 0.1670, 0.2067, 0.1033, 0.1727, 0.0669])

plt.figure(figsize=(10, 6))

# Accuracy
plt.plot(years, acc_mean, '-o', label='Accuracy')
plt.fill_between(years, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)

# F1-score
plt.plot(years, f1_mean, '-s', label='F1-score')
plt.fill_between(years, f1_mean - f1_std, f1_mean + f1_std, alpha=0.2)

# AUC
plt.plot(years, auc_mean, '-^', label='AUC')
plt.fill_between(years, auc_mean - auc_std, auc_mean + auc_std, alpha=0.2)

plt.xlabel('Prediction Year')
plt.ylabel('Metric Value')
plt.title('Prediction Year vs Metric with Std Deviation Bands')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # === Provide paths to your 3 CSVs ===
# csv_paths = [
#     r"D:\MS Computer Engineering\Thesis\Code\AMD\Dense_Results_Yearwise_CV_20250615_214628\Fold1\test_predictions.csv",
#     r"D:\MS Computer Engineering\Thesis\Code\AMD\Dense_Results_Yearwise_CV_20250615_214628\Fold2\test_predictions.csv",
#     r"D:\MS Computer Engineering\Thesis\Code\AMD\Dense_Results_Yearwise_CV_20250615_214628\Fold3\test_predictions.csv"
# ]

# # ✅ 1️⃣ Load and combine
# all_dfs = [pd.read_csv(p) for p in csv_paths]
# df_all = pd.concat(all_dfs, ignore_index=True)

# print(f"✅ Combined data shape: {df_all.shape}")
# print("✅ Available columns:", list(df_all.columns))

# print(f"✅ Combined data shape: {df_all.shape}")

# # ✅ 2️⃣ Extract labels
# y_true = df_all["True"]
# y_pred = df_all["Pred"]

# # ✅ 3️⃣ Compute confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# print("\n✅ Confusion Matrix (aggregated over all folds):\n")
# print(cm)

# # ✅ 4️⃣ Plot and save
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap="Blues")
# plt.title("Aggregated Confusion Matrix Across All Folds")
# plt.tight_layout()
# plt.savefig("Aggregated_Confusion_Matrix_All_Folds.png")
# plt.show()

# print("\n✅ Saved aggregated confusion matrix as: Aggregated_Confusion_Matrix_All_Folds.png")

# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # === CHANGE THIS PATH ===
# csv_path = r"D:\MS Computer Engineering\Thesis\Code\AMD\Denseyearimagewise\Year_Wise_Classification_Fold3\test_predictions_20250628_073610.csv"

# # ✅ Load CSV
# df = pd.read_csv(csv_path)

# # ✅ Extract columns
# y_true = df['TrueLabel']
# y_pred = df['PredLabel']
# y_prob = df['Probability']

# # ✅ Compute metrics
# acc = accuracy_score(y_true, y_pred)
# prec = precision_score(y_true, y_pred)
# rec = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# auc = roc_auc_score(y_true, y_prob)

# # ✅ Print nicely
# print("\n✅ Test Metrics from Predictions CSV:")
# print(f"Accuracy:  {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall:    {rec:.4f}")
# print(f"F1 Score:  {f1:.4f}")
# print(f"AUC:       {auc:.4f}")

# # ✅ Optionally save to text file
# result_lines = [
#     "=== TEST METRICS ===",
#     f"Accuracy:  {acc:.4f}",
#     f"Precision: {prec:.4f}",
#     f"Recall:    {rec:.4f}",
#     f"F1 Score:  {f1:.4f}",
#     f"AUC:       {auc:.4f}",
# ]

# output_txt = csv_path.replace(".csv", "_metrics.txt")
# with open(output_txt, "w") as f:
#     for line in result_lines:
#         f.write(line + "\n")

# print(f"\n✅ Metrics also saved to: {output_txt}")