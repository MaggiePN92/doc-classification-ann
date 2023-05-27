import warnings
warnings.filterwarnings("ignore")

# import gzip
import sys
import pickle
from logger import Logger
import time
import pandas as pd
from data.dataset import SignalDataset
import torch
from train_loop import train_model
from data.datasplit import split_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from models import ann, ann_two_layers, \
    ann_three_layer, ann_four_layers, ann_five_layers
from models import ann_test, ann_two_layers_test, \
    ann_three_layer_test, ann_four_layers_test, ann_five_layers_test
from customVectorizers.preprosessor import preprosessor_num, preprosessor_num_and_len, remove_tags_1, remove_tags_2, preprosessor_len, preprosessor_POS_count


def main(path2signaldata : str, device : str = "cuda") -> None:
    """Trains vectorizers and models on the signal dataset in
    a double for-loop. First iterates list of vectorizers, then
    list of models. This means each model is trained as many 
    times as there are vectorizers.  

    Args:
        path2signaldata (str): path to signal dataset.
        device (str): which device to use, gpu or cuda. 
    """
    f1_macro = 0

    """
    df = pd.read_csv(
        path2signaldata,
        delimiter="\t"
    )
    """
    # loading data from zip-file
    df = pd.read_csv(
        path2signaldata, 
        compression='gzip',
        delimiter="\t"
    )

    print("spliting data..")
    train, val = split_data(df)

    # vectorizers with different feature composures
    print("creating vectorizers")

    vecs = [
        ("bigram_len_vec", CountVectorizer(
        preprocessor = preprosessor_len, max_features = 5000, ngram_range=(1,2))),
        ("vanilla__len_vec", CountVectorizer(
        preprocessor = preprosessor_len, max_features = 5000)),
        ("vanilla_vec", CountVectorizer(
        max_features = 5000)),
    ]
    
    """
    best_vecs = [
        ("trigram_len_vec", CountVectorizer(
        preprocessor = preprosessor_len, max_df=0.9999, max_features=7500, ngram_range=(1,3))),
        ("trgram_pos_vec", CountVectorizer(
        preprocessor = preprosessor_POS_count, max_features = 7500, ngram_range=(1,3))),
        ("bigram_len_vec", CountVectorizer(
        preprocessor = preprosessor_len, max_features = 7500, ngram_range=(1,2))),
        ("bigram_pos_vec", CountVectorizer(
        preprocessor = preprosessor_POS_count, max_features = 7500, ngram_range=(1,2))),
    ]


    
    # archived vectorizers...
    archived_vecs = [
        ("tfidf_bigram_vec", TfidfVectorizer(
        preprocessor = preprosessor_len, max_df=0.9999, max_features=7500, ngram_range=(1,2))),
        ("vanilla_and_len_vec", CountVectorizer(
        preprocessor = preprosessor_len, max_features = 7500)),
        ("vanilla_vec_min", CountVectorizer(min_df = 200)),
        ("num_colapsed_and_len_added_vec", CountVectorizer(
        preprocessor = preprosessor_num_and_len, max_features = 7500)),
        ("vanilla_vec", CountVectorizer(max_features = 7500)),
        ("vanilla_vec_few", CountVectorizer(max_df=0.99, max_features = 2500)),
        ("vanilla_vec_frequent", CountVectorizer(min_df=0.05)),
        ("vanilla_vec_rare", CountVectorizer(max_df=0.80, min_df=0.000001)), #
        ("tfidf_bigram", TfidfVectorizer(max_df=0.9999, max_features=5000, ngram_range=(1,2))),
        ("tfidf_bigram_frequent", TfidfVectorizer(max_df=0.99999,
            min_df=0.05, ngram_range=(1,2))),
        ("tfidf_bigram_rare", TfidfVectorizer(max_df=0.8,
            min_df=0.000001, ngram_range=(1,2))),
        ("noun_verbs_only_vec", CountVectorizer(
        preprocessor = remove_tags_1, max_features = 5000)),
        ("noun_verbs_adj_only_vec", CountVectorizer(
        preprocessor = remove_tags_2, max_features = 5000)),
        ("num_colapsed_vec", CountVectorizer(
        preprocessor = preprosessor_num, max_features = 5000)),
        ("tfidf_vec", TfidfVectorizer(max_features = 5000)),
        ("num_colapsed_and_len_added_vec", CountVectorizer(
        preprocessor = preprosessor_num_and_len, max_features = 5000)),
    ]
    """
    

    # list of uninitiated models
    """
    mods2train = [
        ("one_hidden_layer_1024", ann_test.OneHiddenLayer),
        ("two_hidden_layers_1024", ann_two_layers_test.TwoHiddenLayers),
        ("three_hidden_layers_1024", ann_three_layer_test.ThreeHiddenLayers),
        ("four_hidden_layers_1024", ann_four_layers_test.FourHiddenLayers),
        ("five_hidden_layers_1024", ann_five_layers_test.FiveHiddenLayers)
    ]
    """
    # models are put in a list which is iterated
    mods2train = [
        ("one_hidden_layer", ann.OneHiddenLayer),
        ("two_hidden_layers", ann_two_layers.TwoHiddenLayers),
        ("three_hidden_layers", ann_three_layer.ThreeHiddenLayers),
        ("four_hidden_layers", ann_four_layers.FourHiddenLayers),
        ("five_hidden_layers", ann_five_layers.FiveHiddenLayers)
    ]
    
    # start iteration of vectorizers
    for vec_name, vec in vecs:
        print("*"*20)
        print(f"Using {vec_name} to prepare the data.")
        # fit vectorizer to text
        vec.fit(train["text"])
        
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
        print("Creating DataLoader")
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=128, # prev 128, 32
            shuffle=True,
            num_workers=4
        )
        # this is needed to instantiate the models as n_feat 
        # (# features) will be equal to in paramaters 
        n_feat = train_ds[0][0].numel()
        # n_classes will be equall to number of out parameters 
        n_classes = train_ds[0][1].numel()
        print("Data prepped.")
        print(f"# features = {n_feat}")
        print("Initiating training.")

        # iterate models
        for model_name, untrained_model in mods2train:
            print("#"*20)
            print(f"training model: {model_name}.")
            # clear cuda cache
            torch.cuda.empty_cache()
            # instantiate model 
            model = untrained_model(
                n_features = n_feat,
                n_classes = n_classes
            )
            # set model to correct device, gpu recommended
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
            print(f"training time = {training_time}")
            # retrieve f1-macro score from classifcation report
            curr_f_score = clf_report["macro avg"]["f1-score"]
            print(f"macro average f1-score = {curr_f_score}")

            # log results to file
            model_log = Logger(
                f"{model_name}-{vec_name}",
                training_time,
                curr_f_score,
                accuracy,
                clf_report,
                accum_train_loss,
                accum_val_loss
            )
            model_log.log()
            # if this model has higher f1-score than previous, set
            # this model to best model together with its vectorizer
            if curr_f_score > f1_macro:
                best_model = trained_model
                best_vec = vec
                best_vec_name = vec_name
                best_model_repr = f"{model_name}-{vec_name}-{curr_f_score:.2f}"
                f1_macro = curr_f_score

    print("Best performing model:", best_model_repr)
    print("with macro f-score =", f1_macro)
    # saving model state dict to disk
    torch.save(best_model.state_dict(), f"output/{best_model_repr}.bin")
    # saving best vect as pickle-file
    with open(f"output/{best_vec_name}", "wb") as f:
        pickle.dump(best_vec, f)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Please enter the path to the training data, and name of device")
    
    else:
        # reads filepath from command line
        path_to_file = sys.argv[1]
        device = sys.argv[2]

        main(path_to_file, device = device)
