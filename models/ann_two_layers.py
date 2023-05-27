from torch import nn 


class TwoHiddenLayers(nn.Module):
    def __init__(
        self,
        n_features : int,
        n_classes : int,
        h1_in : int = 512,
        h1_out : int = 256,
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
