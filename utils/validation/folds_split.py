import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data_on_folds(annotations_file, num_folds=5, random_state=808):
    """_summary_

    Args:
        annotations_file (str): _description_
        num_folds (int, optional): Number of folds to split the data.
        Defaults to 5.
        random_state (int, optional): Defaults to 808.

    Returns:
        Dict: The dictionary with train and val indices.
    """

    # Read the annotation CSV file
    annotations_df = pd.read_csv(annotations_file)
    # The second column contains the labels
    labels = annotations_df.iloc[:, 1]

    # Perform stratified train-val split based on the labels
    skf = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=random_state
    )
    train_indices, val_indices = next(skf.split(annotations_df, labels))

    # Create the folds dictionary
    folds = {
        "train": train_indices,
        "val": val_indices,
    }

    return folds
