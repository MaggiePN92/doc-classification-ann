from torch import nn 

# Format: [num_hidden_nodes_per_layer]  <vectorizer_name>  
#                                       (num_features)  = f1-score  
# missing f1-scores might occurr for runs where an error ocurred...

# Hidden nodes [128, 64]: Vanilla_vec (5000) = 0.444
# Hidden nodes [128, 64]: tfidf_vec (5000) = 0.433
# Hidden nodes [128, 64]: vanilla_vec_few (2500) = 0.409
# Hidden nodes [128, 64]: vanilla_vec_frequent (501) = 0.275
# Hidden nodes [128, 64]: "vanilla_vec_rare"
# Hidden nodes [128, 64]:"vanilla_bigram", (5000) = 0.457
# Hidden nodes [128, 64]:"tfidf_bigram", (5000) = 0.460
# Hidden nodes [128, 64]:"tfidf_bigram_frequent" (515) = 0.294
# Hidden nodes [128, 64]:"tfidf_bigram_rare"
# Hidden nodes [128, 64]:"noun_verbs_only_vec" (5000) = 0.243
# Hidden nodes [128, 64]:"noun_verbs_adj_only_vec" (5000) = 0.233
# Hidden nodes [128, 64]:"num_colapsed_vec" (5000) = 0.436
# Hidden nodes [128, 64]:"num_colapsed_and_len_added_vec" (5000) = 0.445

# Hidden nodes [512, 256] "vanilla_vec" (5000) = 0.471
# Hidden nodes [512, 256] "vanilla_and_len_vec" (5000) = 0.475
# Hidden nodes [512, 256] "bigram_len_vec" (5000) = 0.497
# Hidden nodes [512, 256] "tfidf_bigram_vec" (5000) = 0.489
# Hidden nodes [512, 256] "num_colapsed_and_len_added_vec" (5000) = 0.471

# models below have lr = 0,01
# [1024, 512] for all below
# Hidden nodes [1024] "vanilla_vec" (7500) = 0.480
# Hidden nodes [1024] "vanilla_vec_min" (6485) = 0.488
# Hidden nodes [1024] "vanilla_and_len_vec"
# Hidden nodes [1024] "bigram_len_vec" (7500) = 0.504
# Hidden nodes [1024] "bigram_pos_vec"
# Hidden nodes [1024] "tfidf_bigram_vec" (7500) = 0.496
# Hidden nodes [1024] "num_colapsed_and_len_added_vec" (7500) = 0.486

# Hidden nodes [1024] "trgram_pos_vec" (7500) = 0.509
# Hidden nodes [1024] "trigram_len_vec" (7500) = 0.513
# Hidden nodes [1024] "tfidf_bigram_vec" (7500) = 0.503
# Hidden nodes [1024] "vanilla_and_len_vec" (7500) = 0.481
# Hidden nodes [1024] "bigram_len_vec" (7500) = 0.506
# Hidden nodes [1024] "bigram_pos_vec" (7500) = 0.507


class TwoHiddenLayers(nn.Module):
    def __init__(
        self,
        n_features : int,
        n_classes : int,
        h1_in : int = 1024,
        h1_out : int = 512,
    ):
        """Neural network with two hidden layers. 

        Args:
            n_features (int): How many features are present in the 
            dataset. This corresponds to tokens in the dataset. 
            n_classes (int): How many classes are in the dataset.
            This corresponds to websites in the signal dataset. 
            h1_params (int): how many parameters in the hidden layer. 
            More parameters means higher complexity and longer training
            and inference time. 
            h1_out (int): # params out from first hidden layer. Is equal 
            to # in params in second hidden layer. 
        """
        super().__init__()
        # input -> first hidden layer
        self.linear1 = nn.Linear(n_features , h1_in, bias=True)
        # batch norm
        self.batchnorm1 = nn.BatchNorm1d(h1_in)
        # first hidden layer -> output 
        self.linear2 = nn.Linear(h1_in , h1_out, bias=True)
        # batch norm
        self.batchnorm2 = nn.BatchNorm1d(h1_out)
        # second hidden layer -> output 
        self.linear3 = nn.Linear(h1_out , n_classes, bias=True)
        # squash out to probs w/ sigmoid
        self.sigmoid = nn.Sigmoid()
        # ReLU (= max(0, x)) set at activation function
        self.relu = nn.ReLU()
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
        # drop out applied to reduce overfitting 
        # drou out prob = 0.5 
        x = self.dropout50(x)
        # output is fed to third layer
        x = self.linear2(x)
        x = self.batchnorm2(x)
        # activation func. 
        x = self.relu(x)
        x = self.dropout25(x)
        # out fed to output layer
        x = self.linear3(x)
        # output is squashed into probablities 
        preds = self.sigmoid(x)
        return preds.squeeze(-1)


def main():
    """For testing purposes."""
    import torch

    test_mod = TwoHiddenLayers(
        100, 
        10,
        20, 
        20,
    )

    t = torch.rand((32, 100))

    out = test_mod(t)
    print(out.shape)


if __name__ == "__main__":
    main()
