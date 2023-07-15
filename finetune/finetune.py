import torch
import torch.optim as optim
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    loss_fn,
    learning_rate,
    logger=None,
):
    """Train the model using the provided data loaders.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        num_epochs (int): The number of training epochs.
        loss_fn (callable): The loss function to use.
        learning_rate (float): The learning rate for the optimizer.
        logger (logging.Logger, optional): The logger object for logging
        progress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    with tqdm(total=num_epochs, desc="Epochs", dynamic_ncols=True) as epoch_bar:
        for epoch in range(num_epochs):
            # Set model to training mode
            model.train()

            # Initialize progress bar for training batches
            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs} (Train)",
                dynamic_ncols=True,
            ) as train_batch_bar:
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Update the training batch progress bar
                    train_batch_bar.update(1)
                    train_batch_bar.set_postfix({"Train Loss": loss.item()})

            # Evaluate on validation set
            model.eval()
            val_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Initialize progress bar for validation batches
            with tqdm(
                total=len(val_loader),
                desc=f"Epoch {epoch+1}/{num_epochs} (Val)",
                dynamic_ncols=True,
            ) as val_batch_bar:
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)

                        val_loss += loss_fn(outputs, labels).item()
                        total_correct += (predicted == labels).sum().item()
                        total_samples += labels.size(0)

                        # Update the validation batch progress bar
                        val_batch_bar.update(1)

            # Compute validation metrics
            val_loss /= len(val_loader)
            val_accuracy = total_correct / total_samples

            # Log training progress using the logger
            if logger:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )

            # Update the epoch progress bar
            epoch_bar.update(1)
