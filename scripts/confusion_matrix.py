import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# True Positive, False Positive, True Negative, False Negative counts
TP, FP, TN, FN = 409, 462, 382, 66

# Construct the confusion matrix
confusion_mat = np.array([[TP, FP], [FN, TN]])

# Calculate percentages
percentages = (confusion_mat / np.sum(confusion_mat)) * 100

# Plot confusion matrix with Seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.6, f'{percentages[i, j]:.2f}%', 
                 ha='center', va='center', color='black')

plt.title('Confusion Matrix for metacognitive evaluation')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
