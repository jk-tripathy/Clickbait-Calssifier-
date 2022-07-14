import matplotlib.pyplot as plt
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve, roc_auc_score

test_results = []
with open('outputs/base/submit_results.tsv') as file:
    rd = csv.reader(file, delimiter="\t")
    next(rd, None)
    for row in rd:
        data = [i for i in row]
        test_results.append(int(data[1]))

actual = []
with open('data/CoLA/test_labels.tsv') as file:
    rd = csv.reader(file, delimiter="\t")
    next(rd, None)
    for row in rd:
        data = [i for i in row]
        actual.append(int(data[1]))

print(f'accuracy score: {accuracy_score(actual, test_results)}\n')
print(f'confusion matrix:\n{confusion_matrix(actual, test_results   )}\n')
print(
    f'precision score: {precision_score(actual, test_results, average="weighted", zero_division="warn")}\n')
print(
    f'recall score: {recall_score(actual, test_results, average="weighted", zero_division="warn")}\n')
print(f'f1 score: {f1_score(actual, test_results, average="weighted", zero_division="warn")}\n')

lr_precision, lr_recall, _ = precision_recall_curve(actual, test_results)
lr_auc = auc(lr_recall, lr_precision, )
actual_pstvs = []
for ele in actual:
    if ele == 1:
        actual_pstvs.append(1)
no_skill = len(actual_pstvs) / len(actual)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', )
plt.plot(lr_recall, lr_precision, marker='.', label='Albert Fine-Tuning')
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Precision-Recall Curve', fontsize=20)
plt.legend(loc="lower left")
plt.show()



# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(actual, test_results)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=20)
plt.legend(loc="lower right")
plt.show()
