def test_model(model, dataloader, size, criterion, num_epochs, device):
    model.eval()
    all_epoch_loss = []
    all_epoch_acc = []

    # Loop over all epochs
    for epoch in range(num_epochs):
        predictions = np.zeros(size)
        all_classes = np.zeros(size)
        all_proba = np.zeros((size, num_classes))

        i = 0
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, classes in dataloader:
                inputs = inputs.to(device)
                classes = classes.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, classes)
                _, preds = torch.max(outputs.data, 1)

                running_loss += loss.item()
                running_corrects += torch.sum(preds == classes.data)

                # Store predictions, true labels, and probabilities
                batch_size = len(classes)
                predictions[i:i+batch_size] = preds.cpu().numpy()
                all_classes[i:i+batch_size] = classes.cpu().numpy()
                all_proba[i:i+batch_size, :] = outputs.cpu().numpy()

                i += batch_size

        # Calculate epoch loss and epoch accuracy
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.item() / size

        # Print loss and accuracy for each epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Store epoch loss and accuracy
        all_epoch_loss.append(epoch_loss)
        all_epoch_acc.append(epoch_acc)

    return predictions, all_proba, all_classes, all_epoch_loss, all_epoch_acc
