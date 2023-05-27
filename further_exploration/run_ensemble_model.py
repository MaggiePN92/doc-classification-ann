import argparse
import torch
import pickle
from data.dataset import SignalDataset
import pandas as pd
from eval_fns import clf_report
from typing import Dict, Any
from pathlib import Path
from models.ann_four_layers import FourHiddenLayers


def eval_ensemble_on_test(
    path2data : str, device="cpu"
) -> Dict[str, Any]:
    """Function for evaluating model on the test set. Please make sure 
    mapping.json is included in your local directory.

    Args:
        path2data (str): path to test data. 
        device (str, optional): what device to use for computation. Defaults 
        to "cpu".

    Returns:
        Dict[str, Any]: classification report. 
    """
    models_list = []
    path2statedicts = []
    path2vecs = []

    vec_model_path_list = [(path2vecs[i], models_list[i], path2statedicts[i]) for i in range(path2models)]

    # load dataset, assumes delimiter is tab
    df = pd.read_csv(
        path2data, 
        delimiter="\t"
    )

    loaded_models = []
    for path2vec, models, path2dict in vec_model_list:
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
        
        raise NotImplementedError("Define model to use.")
        model = models(n_feat, n_classes)
        # load model
        model.load_state_dict(torch.load(path2model))
        model.to(device)

        # ads model to list
        loaded_models.append(model)
    
    ensemble_model = EnsembleModel([model1, model2, model3])
    ensemble_model.to(device)
    # put model in eval mode
    ensemble_model.eval()
    # make prediciton on validation data and put on correct device
    preds_val = ensemble_model(dataset.txt.to(device))
    # make prediciton on validation data and put on correct device
    y_true_val = dataset.classes.to(device)
    # make classification report
    clf_rep = clf_report(
        preds_val, y_true_val, list(dataset.mapping.keys())
    )
    return clf_rep


if __name__ == "__main__":
    mapping = Path("mapping.json")
    if not mapping.exists():
        raise Exception(
            "Make sure mapping.json is present in your working directory. This file is needed to map classes to correct integers in the dataset."
        )
    # parse cli arguments 
    parser = argparse.ArgumentParser("eval_ensemble_on_test")
    parser.add_argument("path2data", help="Path to test data.", type=str)
    args = parser.parse_args()
    
    eval_ensemble_on_test(args.path2data)
