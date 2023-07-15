import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data_on_folds(annotations_file, num_folds=5, random_state=808):
    """Method that splits data on folds using stratification.

    Args:
        annotations_file (str): Path to the annotation CSV file.
        num_folds (int, optional): Number of folds to split the data.
            Defaults to 5.
        random_state (int, optional): Random state for reproducibility.
            Defaults to 808.

    Returns:
        list: of dictionaries with train and validation indices.
    """
    # Read the annotation CSV file
    annotations_df = pd.read_csv(annotations_file)
    # The second column contains the labels
    labels = annotations_df.iloc[:, 1]

    # Perform stratified train-val split based on the labels
    skf = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=random_state
    )

    folds = []

    # Iterate over the folds and obtain the train and val indices
    for train_indices, val_indices in skf.split(annotations_df, labels):
        folds.append({"train": train_indices, "val": val_indices})

    return folds
