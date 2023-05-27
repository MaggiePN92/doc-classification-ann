import argparse
import pickle
import numpy as np
import torch
from data.dataset import SignalDataset
from data.datasplit import split_data
import time
from train_loop import train_model
import pandas as pd

# for testing 
import models.ann_three_layer_test

def main(
    path2signaldata : str, path2vec : str, device="cuda"
) -> None:
    """Retrains best model three times and measures mean
    and standard deviation of metrics results are written to text file.

    Args:
        path2signaldata (str): path to dataset. 
        path2vec (str): path to already fitted vectorizer. 
        device (str, optional): what device to use. Defaults to "cuda".
    """
    # define what model to train and with what vectorizer
    # train model three times and measure average and standard
    # deviation for:
    #   - accuracy
    #   - precision, recall and F1-macro score 
    # results are written to text file
    
    acc = []
    precision = []
    recall = []
    f1 = []

    print(f"Reading data from {path2signaldata}.")
    df = pd.read_csv(
        path2signaldata,
        compression='gzip',
        delimiter="\t"
    )
    train, val = split_data(df)

    # load pretrained vectorizer 
    with open(path2vec, "rb") as f:
        vec = pickle.load(f)
    
    # make training dataset
    train_ds = SignalDataset(
        train["text"],
        train["source"],
        vec
    )
    # make validation dataset
    val_ds = SignalDataset(
        val["text"],
        val["source"],
        vec
    )
    # make training dataloader 
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )
    # this is needed to instantiate the models as n_feat 
    # (# features) will be equal to in paramaters 
    n_feat = train_ds[0][0].numel()
    # n_classes will be equall to number of out parameters 
    n_classes = train_ds[0][1].numel()
    print("Data prepped.")

    #raise NotImplementedError("Define model to train.")
    # for testing 
    untrained_model = models.ann_three_layer_test.ThreeHiddenLayers

    for i in range(1,4):
        print(f"Iteration {i}")
        # clear cuda cache
        torch.cuda.empty_cache()
        # instantiate model 
        model = untrained_model(
            n_features = n_feat,
            n_classes = n_classes
        )
        model.to(device)
    
        # set AdamW as optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.05
        )
        # use ExponetialLR as learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)

        # train model and measure training time
        start_time = time.time()
        trained_model, clf_report, accuracy, accum_train_loss, accum_val_loss = train_model(
            model = model, 
            n_epochs = 100, 
            train_dataloader = train_dataloader,
            val_dataset = val_ds,
            optim = optimizer, 
            scheduler = scheduler,
            device = device,
        )
        training_time = time.time() - start_time

        print(f"Model {i} done training - ", end="")
        print(f"training time = {training_time}")

        # retrieve metrics from classification report
        precision.append(clf_report["macro avg"]["precision"])
        recall.append(clf_report["macro avg"]["recall"])
        f1.append(clf_report["macro avg"]["f1-score"])
        acc.append(accuracy)

    # calculate mean and std
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    f1_mean = np.mean(f1)
    acc_mean = np.mean(acc)

    precision_std = np.std(precision)
    recall_std = np.std(recall)
    f1_std = np.std(f1)
    acc_std = np.std(acc)

    # write results to text file
    with open("output/report_1.txt", "w") as f:
        f.write(f"f1: mean={f1_mean}; std={f1_std};")
        f.write(f"recall: mean={recall_mean}; std={recall_std};")
        f.write(f"precision: mean={precision_mean}; std={precision_std};")
        f.write(f"acc: mean={acc_mean}; std={acc_std};")


if __name__ == "__main__":
    # parse CL args
    parser = argparse.ArgumentParser("retrain_best_mod")
    parser.add_argument("path2data", help="Path to test data.", type=str)
    parser.add_argument("path2vec", help="Path to vectorizer.", type=str)
    parser.add_argument(
        "device", 
        help="What device to use, cpu or gpu.", 
        type=str,
        default="cpu"
    )
    args = parser.parse_args()
    
    main(args.path2data, args.path2vec, args.device)
