from sklearn.metrics import (confusion_matrix ,classification_report , roc_auc_score)
from sklearn.preprocessing import label_binarize

def compute_metrics(y_true,  y_pred, y_probs):
    cm = confusion_matrix(y_true , y_pred)
    report = classification_report(y_true , y_pred)

    y_true_bin = label_binarize(y_true , classes=[0,1,2])
    auc = roc_auc_score(y_true_bin , y_probs)

    return cm , report , auc 