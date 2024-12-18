# evaluate.py

import torch
def evaluate_model(model, testloader, DEVICE):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # No need for gradients during evaluation
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Get predictions (highest logit value)
            _, predicted = torch.max(outputs.data, 1)

            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print accuracy
    accuracy = 100 * correct / total
    
    return accuracy