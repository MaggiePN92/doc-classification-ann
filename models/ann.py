from torch import nn 


class OneHiddenLayer(nn.Module):
    def __init__(
        self,
        n_features : int,
        n_classes : int,
        h1_params : int = 512
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
