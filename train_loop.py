import torch
from torch.nn import functional as F
from typing import Any
from eval_fns import evaluate, clf_report
from eary_stopping import EarlyStopping


def train_model(
    model : Any, 
    n_epochs: int, 
    train_dataloader : torch.utils.data.DataLoader,
    val_dataset : Any,
    optim : Any, 
    scheduler : Any,
    device : str,
    loss_fn = F.binary_cross_entropy
):
    """Defines the training loop for each model. 

    Args:
        model (Any): Model to be trained.
        n_epochs (int): how many epochs should the model be trained for. 
        train_dataloader (torch.utils.data.DataLoader): dataloader with training data. 
        val_dataset (Any): dataset used to validate model. 
        optim (Any): opitmizer.
        scheduler (Any): learning rate schedulizer.
        device (str): what device to use, should be "cuda" or "cpu". 
        loss_fn (_type_, optional): loss function to optimize. Defaults to F.binary_cross_entropy.

    Returns:
        Any: trained model. 
    """
    early_stopper = EarlyStopping()
    accum_train_loss = []
    accum_val_loss = []
    prev_f1score = 0.0

    for epoch in range(n_epochs):
        # put model in train mode, if drop out is included in forward this will be activated
        model.train()
        # get data and targets from the dataloader, these are put to the correct device
        for inputs, classes in train_dataloader:
            inputs, classes = inputs.to(device), classes.to(device)
            # zero out gradients
            optim.zero_grad()
            # make prediction 
            y_pred = model(inputs)
            # calcualte loss
            loss = loss_fn(
                y_pred, 
                classes
            ).mean()
            # calculate grads 
            loss.backward()
            # update weights w.r.t. grads 
            optim.step()
        # adjust learning rate
        scheduler.step()

        print(f"Epoch {epoch} completed.")

        with torch.no_grad():
            model.eval()
            # make prediciton on validation data and put on correct device
            preds_val = model(val_dataset.txt.to(device))
            # make prediciton on validation data and put on correct device
            y_true_val = val_dataset.classes.to(device)
            val_loss = loss_fn(y_pred, classes).mean()
            accum_val_loss.append(val_loss.item())
            clf_rep = clf_report(
                preds_val, y_true_val, list(val_dataset.mapping.keys())
            )
            
            # check if model has improved - if yes best state dict is set
            # to current state dict. The best state dict will be the models
            # final form.
            curr_f1score = clf_rep['macro avg']['f1-score']
            if curr_f1score > prev_f1score:
                best_state = model.state_dict()
                prev_f1score = curr_f1score
                best_clf_rep = clf_rep
            
            print(f"Current f1: {curr_f1score}")
        
        accum_train_loss.append(loss.item())
        if early_stopper.early_stop(clf_rep['macro avg']['f1-score']):
            break
    
    # the best state dict is set to the models current state dict
    model.load_state_dict(best_state)

    # when we want to evaluate our model we do not need to calculate grads, nor
    # do we want dropout to be activated.
    with torch.no_grad():
        model.eval()
        # make prediciton on validation data and put on correct device
        preds_val = model(val_dataset.txt.to(device))
        # make prediciton on validation data and put on correct device
        y_true_val = val_dataset.classes.to(device)
        # call evaluation function and get loss, accuracy and classification
        # report 
        valid_loss, valid_accuracy = evaluate(
            preds_val, y_true_val, loss_fn
        )

    print(f"Val loss: {valid_loss} - Val acc: {valid_accuracy} - Val f1: {prev_f1score }")
    
    return model, best_clf_rep, valid_accuracy, accum_train_loss, accum_val_loss
