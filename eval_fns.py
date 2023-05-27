from sklearn.metrics import classification_report
from typing import List, Dict, Any


def evaluate(
    y_preds, y_true, loss_fn
):
    """Gathers all functions that are used to evaluate the model
    performance. 

    Args:
        y_preds (torch.Tensor): predictions from the model. 
        y_true (torch.Tensor): gold labels. 
        loss_fn (Callable): loss function which takes y_preds and 
        y_true as args. 
        label_names (List[str]): list with name of the labels

    Returns:
        Tuple[float, float, Dict]: results from evaluation. 
    """
    # calculate loss 
    loss = loss_fn(
        y_preds, y_true
    ).item()
    
    # calculate accuracy
    acc = accuracy(
        y_preds, 
        y_true
    ).item()

    return loss, acc


def accuracy(y_preds, y_true) -> float:
    """_summary_

    Args:
        y_preds (torch.Tensor): predictions from the model. 
        y_true (torch.Tensor): gold labels. 

    Returns:
        float: accuracy of model. 
    """
    # taken from https://github.uio.no/in5550/2023/blob/main/labs/03/session_03.ipynb
    # if value is greather than 0.5, it's set to 1:
    # example: [0.3, 0,15, 0.55] -> [0, 0, 1]
    y_pred = (y_preds > 0.5).long()
    y_true = y_true.long()
    # compares tensors with broadcasting:
    # exmaple: ([1, 2, 4] == [1, 3, 4]) = [True, False, True]
    is_correct = y_pred == y_true
    # calculate mean
    average = is_correct.float().mean()
    return average * 100.0


def clf_report(y_preds, y_true, label_names : List[str]) -> Dict[str, Any]:
    """Uses scikit-learns make_classification method to calculate:
        
        - F1-score
        - precision
        - recall

    for each class. 

    See: 
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

    Args:
        y_preds (torch.Tensor): predictions from the model. 
        y_true (torch.Tensor): gold labels.
        label_names (List[str]): list with name of the labels

    Returns:
        Dict[str, Any]: classification report. 
    """
    # If value is greather than 0.5, it's set to 1:
    # example: [0.3, 0,15, 0.55] -> [0, 0, 1].
    # Send result to cpu so scikit_learn will work. 
    y_pred = (y_preds > 0.5).long().cpu()
    y_true = y_true.long().cpu()
    return classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True
    )
