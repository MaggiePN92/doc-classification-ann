from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data : pd.DataFrame, 
    test_size : float = 0.15, 
    stratify_on : str = "source", 
    random_state : int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified and reproducible split of data. 

    Args:
        data (pd.DataFrame): data to split. 
        test_size (float, optional): size of test in fraction. Defaults to 0.15.
        stratify_on (str, optional): name of column to stratify on. Defaults to "source".
        random_state (int, optional): what random state to use. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    train, test = train_test_split(
        data, 
        test_size = test_size, 
        stratify = data[stratify_on], 
        random_state = random_state
    )
    return train, test
