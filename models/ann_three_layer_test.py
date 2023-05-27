from torch import nn 

# Format: [num_hidden_nodes_per_layer]  <vectorizer_name>  
#                                       (num_features)  = f1-score  
# missing f1-scores might occurr for runs where an error ocurred...

# [128, 64, 128] for all below
# Hidden nodes [128, 64, 128]: Vanilla_vec (5000) = 0.413
# Hidden nodes [128, 64, 128]: tfidf_vec (5000 )= 0.412
# Hidden nodes [128, 64, 128]: vanilla_vec_few (2500) = 0.386
# Hidden nodes [128, 64, 128]: vanilla_vec_frequent (501) = 0.250
# Hidden nodes [128]: "vanilla_vec_rare"
# Hidden nodes [128]:"vanilla_bigram", (5000) = 0.424
# Hidden nodes [128]:"tfidf_bigram",  (5000) = 0.426
# Hidden nodes [128]:"tfidf_bigram_frequent" (515) = 0.259
# Hidden nodes [128]:"tfidf_bigram_rare"
# Hidden nodes [128]:"noun_verbs_only_vec" (5000) = 0.206
# Hidden nodes [128]:"noun_verbs_adj_only_vec" (5000) = 0.214
# Hidden nodes [128]:"num_colapsed_vec"  (5000) = 0.405
# Hidden nodes [128]:"num_colapsed_and_len_added_vec" (5000) = 0.419


# [512, 512,  256] for all below
# Hidden nodes [512, ..., 256] "vanilla_vec" (5000) = 0.472
# Hidden nodes [512] "vanilla_and_len_vec" (5000) = 0.468
# Hidden nodes [512] "bigram_len_vec" (5000) = 0.487
# Hidden nodes [512] "tfidf_bigram_vec" (5000) = 0.483
# Hidden nodes [512] "num_colapsed_and_len_added_vec" (5000) = 0.469

# models below have lr = 0,01
# [1024, 1024, 512] for all below
# Hidden nodes [1024] "vanilla_vec" (7500) = 0.482
# Hidden nodes [1024] "vanilla_vec_min"
# Hidden nodes [1024] "vanilla_and_len_vec"
# Hidden nodes [1024] "bigram_len_vec"  (7500) = 0.506
# Hidden nodes [1024] "bigram_pos_vec"
# Hidden nodes [1024] "tfidf_bigram_vec" (7500) = 0.498
# Hidden nodes [1024] "num_colapsed_and_len_added_vec"

# Hidden nodes [1024] "trgram_pos_vec" (7500) = 0.511
# Hidden nodes [1024] "trigram_len_vec" (7500) = 0.507
# Hidden nodes [1024] "tfidf_bigram_vec" (7500) = 0.497
# Hidden nodes [1024] "vanilla_and_len_vec" (7500) = 0.483
# Hidden nodes [1024] "bigram_len_vec" (7500) = 0.508
# Hidden nodes [1024] "bigram_pos_vec" (7500) = 0.507


class ThreeHiddenLayers(nn.Module):
    def __init__(
        self,
        n_features : int,
        n_classes : int,
        h1_in : int = 1024,
        h1_out : int = 1024,
        h2_out : int = 512,
    ):
        """Neural network with three hidden layers. 

        Args:
            n_features (int): How many features are present in the 
            dataset. This corresponds to tokens in the dataset. 
            n_classes (int): How many classes are in the dataset.
            This corresponds to websites in the signal dataset. 
            h1_params (int): how many parameters in the hidden layer. 
            More parameters means higher complexity and longer training
            and inference time. 
            h1_out (int): # params out from first hidden layer. Has to 
            be equal to # in params in second hidden layer. 
            h2_out (int): # in
        """
        super().__init__()
        # input -> first hidden layer
        self.linear1 = nn.Linear(n_features , h1_in, bias=True)
        # batch norm
        self.batchnorm1 = nn.BatchNorm1d(h1_in)
        # first hidden layer -> second 
        self.linear2 = nn.Linear(h1_in , h1_out, bias=True)
        # batch norm
        self.batchnorm2 = nn.BatchNorm1d(h1_out)
        # second hidden layer -> third 
        self.linear3 = nn.Linear(h1_out , h2_out, bias=True)
        # batch norm
        self.batchnorm3 = nn.BatchNorm1d(h2_out)
        # third hidden layer -> fourth
        self.linear4 = nn.Linear(h2_out , n_classes, bias=True)
        # ReLU (= max(0, x)) set at activation function
        self.relu = nn.ReLU()
        # squash out to probs w/ sigmoid
        self.sigmoid = nn.Sigmoid()
        # dropout with p=0.5 and p=0.25
        self.dropout50 = nn.Dropout(0.5)
        self.dropout25 = nn.Dropout(0.25)

    def forward(self, x):
        """Defines what happens when the model is called, model(x)."""
        # input is fed to the first linear layer
        x = self.linear1(x)
        # batch norm is applied to output
        x  = self.batchnorm1(x)
        # output is fed through actiation func. 
        x = self.relu(x)
        x = self.dropout50(x)
        # output is fed to third layer
        x = self.linear2(x)
        x = self.batchnorm2(x)
        # activation func. 
        x = self.relu(x)
        x = self.dropout50(x)
        # out fed to output layer
        x = self.linear3(x)
        x = self.batchnorm3(x)
        # x -> activation func -> output layer
        x = self.relu(x)
        x = self.dropout25(x)
        x = self.linear4(x)
        # output is squashed into probablities 
        preds = self.sigmoid(x)
        return preds.squeeze(-1)


def main():
    import torch

    test_mod = ThreeHiddenLayers(
        100, 
        10,
        20, 
        20,
        10
    )

    t = torch.rand((32, 100))

    output = test_mod(t)
    print(output.shape)


if __name__ == "__main__":
    main()
