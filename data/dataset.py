import torch
import pandas as pd
import json
from typing import Dict, Any


class SignalDataset(torch.utils.data.Dataset):
    def __init__(
          self, 
          txt : pd.Series, 
          classes_as_str : pd.Series,
          vectorizer : Any
      ):
        """Dataset helper class for the signal dataset. Will vectorize text data
        with vectorizer function. 

        Args:
            txt (pd.Series): A pandas Series where each row is a document from a given
            webpage. 
            classes_as_str (pd.Series): pd.Series where each row is a website. Assumes the class
            name (str) is not mapped to an integer. 
            vectorizer (Any, optional): Can be an instance of 
            sklearn.feature_extraction.text.CountVectorizer. Can also be self made vecorizer which 
            implements a transform method that returns a sparse matrix (or some object that implements
            a toarray method).
        """
        self.vectorizer = vectorizer
        # text is first transformed into bag-of-words, then transformed into an
        # torch float tensor. 
        self.txt = torch.as_tensor(
            vectorizer.transform(txt).toarray()
        ).float()
        # classes_as_str is a pandas.Series like this [website1, website2, ...]
        # and needs to be mapped to an integer. We also have to make sure we know
        # what classes are mapped to what integer. This is where sel.mapping is 
        # important. 
        self.classes_as_str = classes_as_str
        # the mapping is read or written if it's not found in the working directory
        self.mapping = self.read_mapping()
        # the str classes are mapped to integers based on the mapping 
        mapped_classes = classes_as_str.replace(self.mapping)
        # after being mapped to integers, we one hot encode the int classes, this means
        # we turn [0,1,2,3] into: 
        # [
        #   [1, 0, 0, 0]  
        #   [0, 1, 0, 0]  
        #   [0, 0, 1, 0]  
        #   [0, 0, 0, 1]  
        # ]
        self.classes = torch.nn.functional.one_hot(
            torch.as_tensor(mapped_classes.tolist())
        ).float()

    def __getitem__(self, idx : int):
        """Helper function that implements indexing in the SignalDataset."""
        x = self.txt[idx]
        y = self.classes[idx]
        return x, y

    def __len__(self):
        """Returns len if len(SignalDataset) is called."""
        return len(self.txt)

    def read_mapping(self) -> Dict[str, int]:
        """Reads mapping for str classes, if mapping.json in working dir. If not 
        the mapping is created and written to woring dir. 

        Returns:
            Dict[str, int]: mapping of str to int. 
        """
        # first try to read mapping.json
        try:
            with open("mapping.json", "r") as f:
                mapping = json.loads(f.read())
        # if file not found we create a mapping 
        except FileNotFoundError:
            print("Did not find mapping.json which maps classes to integers.")
            print("Creating a mapping, and writing it to mapping.json in working directory.")
            
            mapping = {src:i for i, src in enumerate(self.classes_as_str.unique())}
            print(f"mapping={mapping}")

            with open("mapping.json", "w") as f:
                f.write(json.dumps(mapping))

        return mapping
