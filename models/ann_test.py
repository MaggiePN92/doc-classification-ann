from torch import nn 

# Format: [num_hidden_nodes_per_layer]  <vectorizer_name>  
#                                       (num_features)  = f1-score  
# missing f1-scores might occurr for runs where an error ocurred...

# Hidden nodes [128]: Vanilla_vec (5000) = 0.416 
# Hidden nodes [128]: tfidf_vec (5000) = 0.415
# Hidden nodes [128]: vanilla_vec_few (2500) = 0.390
# Hidden nodes [128]: vanilla_vec_frequent (501) = 0.259
# Hidden nodes [128]: "vanilla_vec_rare"
# Hidden nodes [128]:"vanilla_bigram", (5000) = 0.431
# Hidden nodes [128]:"tfidf_bigram", (5000) = 0.438
# Hidden nodes [128]:"tfidf_bigram_frequent" (515) = 0.289
# Hidden nodes [128]:"tfidf_bigram_rare"
# Hidden nodes [128]:"noun_verbs_only_vec" (5000) = 0.210
# Hidden nodes [128]:"noun_verbs_adj_only_vec" (5000) = 0.224
# Hidden nodes [128]:"num_colapsed_vec" (5000) = 0.420
# Hidden nodes [128]:"num_colapsed_and_len_added_vec" (5000) = 0.420

# Hidden nodes [512] "vanilla_vec" (5000) = 0.443
# Hidden nodes [512] "vanilla_and_len_vec" (5000) = 0.446
# Hidden nodes [512] "bigram_len_vec" (5000) = 0.468
# Hidden nodes [512] "tfidf_bigram_vec" (5000) = 0.460
# Hidden nodes [512] "num_colapsed_and_len_added_vec" (5000) = 0.445

# models below have lr = 0,01
# Hidden nodes [1024] "vanilla_vec" (7500) = 0.463
# Hidden nodes [1024] "vanilla_vec_min" (6485) = 0.461
# Hidden nodes [1024] "vanilla_and_len_vec"
# Hidden nodes [1024] "bigram_len_vec" (7500) = 0.485
# Hidden nodes [1024] "bigram_pos_vec"
# Hidden nodes [1024] "tfidf_bigram_vec" (7500) = 0.478
# Hidden nodes [1024] "num_colapsed_and_len_added_vec"  (7500) = 0.460

# Hidden nodes [1024] "trgram_pos_vec" (7500) = 0.494
# Hidden nodes [1024] "trigram_len_vec" (7500) = 0.486
# Hidden nodes [1024] "tfidf_bigram_vec" (7500) = 0.469
# Hidden nodes [1024] "bigram_len_vec" (7500) = 0.486
# Hidden nodes [1024] "bigram_pos_vec" (7500) = 0.491


class OneHiddenLayer(nn.Module):
    def __init__(
        self,
        n_features : int,
        n_classes : int,
        h1_params : int = 1024
    ):
        """Neural network with one hidden layer. 

        Args:
            n_features (int): How many features are present in the 
            dataset. This corresponds to tokens in the dataset. 
            n_classes (int): How many classes are in the dataset.
            This corresponds to websites in the signal dataset. 
            h1_params (int): how many parameters in the hidden layer. 
            More parameters means higher complexity and longer training
             and inference time. 
        """
        super().__init__()
        # input -> first hidden layer
        self.linear1 = nn.Linear(n_features , h1_params, bias=True)
        # ReLU (= max(0, x)) set at activation function
        self.relu = nn.ReLU()
        # first hidden layer -> output 
        self.linear2 = nn.Linear(h1_params , n_classes, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout40 = nn.Dropout(0.40)
        self.dropout15 = nn.Dropout(0.15)
        self.batch_norm = nn.BatchNorm1d(h1_params)

    def forward(self, x):
        """Defines what happens when the model is called, model(x)."""
        # input is fed to the first linear layer
        x = self.linear1(x)
        # batch norm is applied
        x = self.batch_norm(x)
        # output is fed through actiation func. 
        x = self.relu(x)
        # dropout with p=0.15 is applied to reduce overfitting
        x = self.dropout40(x)
        # output is fed to output layer
        x = self.linear2(x)
        # output are squashed into probablities 
        preds = self.sigmoid(x)
        return preds.squeeze(-1)


def main():
    """For testing purposes."""
    import torch

    test_mod = OneHiddenLayer(
        100,
        20,
        10
    )

    t = torch.rand((32, 100))

    output = test_mod(t)
    print(output.shape)


if __name__ == "__main__":
    main()
