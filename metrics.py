import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score


def f1_loss(y, predictions):
    y = y.data.cpu().numpy()
    predictions = predictions.data.cpu().numpy()
    number_of_labels = y.shape[1]
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = np.argsort(predictions, axis=1)

    # the true number of labels for each node
    num_labels = np.sum(y, axis=1)
    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-int(num):].tolist())

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(classes=range(number_of_labels)).fit_transform(pred_reshaped)
    f1_micro = f1_score(y, pred_transformed, average='micro')
    f1_macro = f1_score(y, pred_transformed, average='macro')
    return f1_micro, f1_macro


def BCE_loss(outputs: torch.Tensor, labels: torch.Tensor):
    loss = torch.nn.BCELoss()
    bce = loss(outputs, labels)
    return bce


# def _eval_rocauc(y_pred, y_true):
#     '''
#         compute ROC-AUC and AP score averaged across tasks
#     '''
#     rocauc_list_micro = []
#     rocauc_list_macro = []
#
#     for i in range(y_true.shape[1]):
#         #AUC is only defined when there is at least one positive data.
#         #print(torch.sum(y_true[:, i] == 1))
#         if torch.sum(y_true[:, i] == 1).item() > 0 and torch.sum(y_true[:, i] == 0).item() > 0:
#             is_labeled = y_true[:, i] == y_true[:, i]
#             rocauc_list_micro.append(roc_auc_score(y_true[is_labeled, i].cpu(),
#                                                    y_pred[is_labeled, i].cpu(), average="micro"))
#             rocauc_list_macro.append(roc_auc_score(y_true[is_labeled, i].cpu(),
#                                                    y_pred[is_labeled, i].cpu(), average="macro"))
#     if len(rocauc_list_micro) == 0:
#         raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
#
#     #return {'rocauc': sum(rocauc_list)/len(rocauc_list)}
#     return sum(rocauc_list_micro)/len(rocauc_list_micro), sum(rocauc_list_macro)/len(rocauc_list_macro)

#
# def _eval_rocauc(y_true, y_pred):
#
#     rocauc_micro = roc_auc_score(y_true, y_pred, average="micro")
#     rocauc_macro = roc_auc_score(y_true, y_pred, average="macro")
#
#     return rocauc_micro, rocauc_macro

def _eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC and AP score averaged across tasks
    '''

    y_true = y_true.detach().cpu().numpy()
    #y_pred = y_pred.detach().cpu().numpy()
    rocauc_list = []

    for i in range(y_true.shape[1]):

        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    #return {'rocauc': sum(rocauc_list)/len(rocauc_list)}
    return sum(rocauc_list) / len(rocauc_list)


def ap_score(y_true, y_pred):

    ap_score = average_precision_score(y_true.cpu().detach().numpy(), y_pred)

    return ap_score