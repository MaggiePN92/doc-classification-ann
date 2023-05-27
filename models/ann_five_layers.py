from torch import nn 


class FiveHiddenLayers(nn.Module):
    def __init__(
        self,
        n_features : int,
        n_classes : int,
        h1_in : int = 512,
        h1_out : int = 256,
        h2_out : int = 128,
        h3_out : int = 64,
        h4_out : int = 32 
    ):
        """Neural network with five hidden layers. 

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
            h3_out (int) : # in params
        """
        super().__init__()
        # different linear layers are initialized 
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
        self.linear4 = nn.Linear(h2_out , h3_out, bias=True)
        self.batchnorm4 = nn.BatchNorm1d(h3_out)
        
        self.linear5 = nn.Linear(h3_out , h4_out, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(h4_out)

        self.linear6 = nn.Linear(h4_out , n_classes, bias=True)
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
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout50(x)
        
        # output is fed to third layer
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout50(x)

        # out fed to output layer
        x = self.linear3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout25(x)
        
        x = self.linear4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout25(x)

        x = self.linear5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout50(x)

        x = self.linear6(x)
        # output is squashed into probablities 
        preds = self.sigmoid(x)
        return preds.squeeze(-1)


def main():
    import torch

    test_mod = FiveHiddenLayers(
        100, 
        20,
    )

    t = torch.rand((32, 100))

    output = test_mod(t)
    print(output.shape)
    print(sum(p.numel() for p in test_mod.parameters()))


if __name__ == "__main__":
    main()
