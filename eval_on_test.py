import argparse
import torch
import pickle
from data.dataset import SignalDataset
import pandas as pd
from eval_fns import clf_report, accuracy
from typing import Dict, Any
from pathlib import Path
from models.ann_three_layer_test import ThreeHiddenLayers

def eval_on_test(
    path2model : str, path2data : str, path2vec : str, device="cpu"
) -> Dict[str, Any]:
    """Function for evaluating model on the test set. Please make sure 
    mapping.json is included in your local directory.

    Args:
        path2model (str): path to pytorch model. 
        path2data (str): path to test data. 
        path2vec (str): path to vectorizer. 
        device (str, optional): what device to use for computation. Defaults 
        to "cpu".

    Returns:
        Dict[str, Any]: classification report. 
    """
    
    # load dataset, assumes delimiter is tab
    df = pd.read_csv(
        path2data, 
        delimiter="\t"
    )
    # loading vectorizer, this is need to make bow features. 
    with open(path2vec, "rb") as f:
        pickled_vect = pickle.load(f)
            
    # make training dataset
    dataset = SignalDataset(
        df["text"],
        df["source"],
        pickled_vect
    )

    # this is needed to instantiate the models as n_feat 
    # (# features) will be equal to in paramaters 
    n_feat = dataset[0][0].numel()
    # n_classes will be equall to number of out parameters 
    n_classes = dataset[0][1].numel()
    print("Data prepped.")
    
    model = ThreeHiddenLayers(n_feat, n_classes)
    # load model
    model.load_state_dict(torch.load(path2model))
    model.to(device)
    # put model in eval mode
    model.eval()
    # make prediciton on validation data and put on correct device
    preds_val = model(dataset.txt.to(device))
    # make prediciton on validation data and put on correct device
    y_true_val = dataset.classes.to(device)
    # make classification report
    clf_rep = clf_report(
        preds_val, y_true_val, list(dataset.mapping.keys())
    )
    acc = accuracy(
        preds_val, 
        y_true_val
    ).item()

    curr_f_score = clf_rep["macro avg"]["f1-score"]

    print(f"macro average f1-score = {curr_f_score}")
    print(f"accuracy = {acc}")
    print(clf_rep)
    
    return curr_f_score, acc, clf_rep


if __name__ == "__main__":
    mapping = Path("mapping.json")
    if not mapping.exists():
        raise Exception(
            "Make sure mapping.json is present in your working directory. This file is needed to map classes to correct integers in the dataset."
        )
    # parse cli arguments 
    parser = argparse.ArgumentParser("eval_on_test")
    parser.add_argument(
        "path2model", 
        help="Path to pytorch model.", 
        type=str,
    )
    parser.add_argument(
        "path2data", 
        help="Path to test data.", 
        type=str
    )
    parser.add_argument(
        "--path2vec",
        nargs=1,
        help="Path to vectorizer.", 
        type=str,
        default="/fp/projects01/ec30/magvic_large_files/output/trgram_pos_vec",
        required=False
    )
    args = parser.parse_args()
    
    eval_on_test(args.path2model, args.path2data, args.path2vec)
