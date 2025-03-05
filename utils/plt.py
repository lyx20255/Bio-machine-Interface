import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

history_dict = history.history

loss_keys = [key for key in history_dict.keys() if 'loss' in key and 'val' not in key]
val_loss_keys = [key for key in history_dict.keys() if 'val_loss' in key]

plt.figure(figsize=(10, 6))

for key in loss_keys:
    plt.plot(history_dict[key], label=f'Training {key}')

for key in val_loss_keys:
    plt.plot(history_dict[key], linestyle='--', label=f'Validation {key}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

acc_keys = [key for key in history_dict.keys() if 'accuracy' in key and 'val' not in key]
val_acc_keys = [key for key in history_dict.keys() if 'val_accuracy' in key]

plt.figure(figsize=(10, 6))

for key in acc_keys:
    plt.plot(history_dict[key], label=f'Training {key}')

for key in val_acc_keys:
    plt.plot(history_dict[key], linestyle='--', label=f'Validation {key}')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

y_mc_pred = np.argmax(model.predict(X_test)[0], axis=1)
y_gc_pred = np.argmax(model.predict(X_test)[1], axis=1)
y_ph_pred = np.argmax(model.predict(X_test)[2], axis=1)
y_t_pred = np.argmax(model.predict(X_test)[3], axis=1)

y_mc_true = np.argmax(y_mc_test, axis=1)
y_gc_true = np.argmax(y_gc_test, axis=1)
y_ph_true = np.argmax(y_ph_test, axis=1)
y_t_true = np.argmax(y_t_test, axis=1)

cm_mc = confusion_matrix(y_mc_true, y_mc_pred)
cm_gc = confusion_matrix(y_gc_true, y_gc_pred)
cm_ph = confusion_matrix(y_ph_true, y_ph_pred)
cm_t = confusion_matrix(y_t_true, y_t_pred)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, (cm, title, ax) in enumerate(zip([cm_mc, cm_gc, cm_ph, cm_t], 
                                        ["MC Classification", "GC Classification", "PH Classification", "T Classification"], 
                                        axes.flatten())):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
for i, (y_test, y_pred, name) in enumerate(zip(
    [y_mc_test, y_gc_test, y_ph_test, y_t_test], 
    model.predict(X_test),
    ["MC", "GC", "PH", "T"]
)):
    for j in range(5):  
        fpr, tpr, _ = roc_curve(y_test[:, j], y_pred[:, j])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} Class {j} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multi-Class Classification")
plt.legend()
plt.show()

feature_extractor = Model(inputs=model.input, outputs=flat)
X_features = feature_extractor.predict(X_test)

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_features)

feature_labels = {
    "MC": np.argmax(y_mc_test, axis=1),
    "GC": np.argmax(y_gc_test, axis=1),
    "PH": np.argmax(y_ph_test, axis=1),
    "T": np.argmax(y_t_test, axis=1)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (feature, labels) in zip(axes.flat, feature_labels.items()):
    sns.scatterplot(ax=ax, x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette="tab10", s=10)
    ax.set_title(f"t-SNE visualization of {feature} Categories")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_embedded = pca.fit_transform(X_features)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for ax, (feature, labels) in zip(axes.flat, feature_labels.items()):
    sns.scatterplot(ax=ax, x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette="tab10", s=10)
    ax.set_title(f"PCA visualization of {feature} Categories")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

variables = {
    "MC": y_mc_test,
    "GC": y_gc_test,
    "PH": y_ph_test,
    "T": y_t_test
}

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for ax, (var_name, y_test) in zip(axes.flat, variables.items()):
    y_pred = model.predict(X_test)[list(variables.keys()).index(var_name)]  
    
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{var_name}{i+1} (AUC={roc_auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--')  
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(f"ROC Curve for {var_name} Classification")
    ax.legend()

plt.tight_layout()
plt.show()

y_true_all = np.concatenate([
    np.argmax(y_mc_test, axis=1),
    np.argmax(y_gc_test, axis=1) + 5, 
    np.argmax(y_ph_test, axis=1) + 10, 
    np.argmax(y_t_test, axis=1) + 15
])

y_pred_all = np.concatenate([
    np.argmax(model.predict(X_test)[0], axis=1),
    np.argmax(model.predict(X_test)[1], axis=1) + 5,
    np.argmax(model.predict(X_test)[2], axis=1) + 10,
    np.argmax(model.predict(X_test)[3], axis=1) + 15
])

cm_total = confusion_matrix(y_true_all, y_pred_all)

labels = [f"MC_{i+1}" for i in range(5)] + \
         [f"GC_{i+1}" for i in range(5)] + \
         [f"pH_{i+1}" for i in range(5)] + \
         [f"T_{i+1}" for i in range(5)]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Overall Confusion Matrix with Named Labels")
plt.show()
