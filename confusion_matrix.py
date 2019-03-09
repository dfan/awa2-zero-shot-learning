from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

labels = np.genfromtxt('data/testclasses.txt', dtype='str').tolist()
labels_dict = {}

index = 0
for label in labels:
  labels_dict[label] = index
  index += 1

with open('predictions_best.txt', 'r') as f:
  predicted_classes = []
  true_classes = []
  for line in f:
    true_class, pred_class = line.strip().split(' ')
    true_class = true_class.split('/')[0]
    true_classes.append(true_class)
    predicted_classes.append(pred_class)

confusion_mat = np.array(confusion_matrix(true_classes, predicted_classes, labels))
confusion_mat = confusion_mat / np.sum(confusion_mat,axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_mat)
fig.colorbar(cax)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels,rotation=45, ha='left', rotation_mode='anchor')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
plt.tight_layout()
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('True', fontweight='bold')

if not os.path.exists('figures'):
  os.mkdir('figures')
plt.savefig('figures/confusion_matrix.png', dpi=500)
