import logging
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils.data.dataset import EyesDataset
from utils.validation.folds_split import split_data_on_folds
from models.***.model import ***
from utils.loss.focal_loss import focal_loss
from finetune.finetune import train


def main():
    # Define paths and parameters
    img_dir = "data/images/"
    annotations_file = "data/ground_truth.csv"
    num_folds = 5
    random_state = 808
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001


    # Split data on folds
    folds = split_data_on_folds(annotations_file, num_folds, random_state)

    for fold in range(num_folds):
        logger.info(f"Training on fold {fold+1}")

        # Create train and validation datasets
        train_indices = folds[fold]["train"]
        val_indices = folds[fold]["val"]

        train_dataset = EyesDataset(
            img_dir,
            annotations_file,
            indices=train_indices,
            mode="train",
            task=1,
        )
        
        val_dataset = EyesDataset(
            img_dir,
            annotations_file,
            indices=val_indices,
            mode="train",
            task=1,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        # Create and initialize the model
        model = Model1()  # Replace with your model class and initialization code

        # Train the model
        train(
            model,
            train_loader,
            val_loader,
            num_epochs,
            focal_loss,
            learning_rate,
            save_every_epoch=False,
            save_path="models/saved",
            logger=logger,
        )

        # Save the trained model weights if needed
        # torch.save(model.state_dict(), f"model_fold{fold+1}.pt")

if __name__ == "__main__":
    # Set up logging to save log messages to a file
    logging.basicConfig(
        level=logging.INFO,
        filename='log_file.log',  # Specify the file path
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Create a logger instance
    logger = logging.getLogger(__name__)

    main()