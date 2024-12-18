# main.py

import torch
import os
import argparse
from data_loader import get_data_loaders
from model import CNN
from train import train_model
from evaluate import evaluate_model

def main():

    parser = argparse.ArgumentParser(description='Train or Test the CNN model')
    parser.add_argument('mode', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training/testing (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (default: 0.001)')
    parser.add_argument('--epoch', type=float, default=3, help='Number of epoch (default: 2)')
    args = parser.parse_args()    
    
    MODEL_PATH = "model/cifar_cnn.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Mode: {args.mode}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Epoch: {args.epoch}")
    print(f"Device: {DEVICE}")

    # Load data
    trainloader, testloader = get_data_loaders(args.batch_size)

    # Initialize model
    model = CNN().to(DEVICE)

    # Train the model
    if args.mode == 'train':

        if os.path.exists(MODEL_PATH):
            print(f"Model already exists at '{MODEL_PATH}'. Skipping training.")
        else:
            print("Starting training...")
            train_model(model, trainloader, MODEL_PATH, DEVICE, args.epoch)
            # Save the trained model
            # DEVICE, NUM_EPOCHS
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to '{MODEL_PATH}'.")
    
    elif args.mode == 'test':
        print("Evaluating the model...")
        # Load the trained model weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        accuracy = evaluate_model(model, testloader, DEVICE)
        print(f'Accuracy of the network on the test dataset: {accuracy:.2f}%')


if __name__ == '__main__':
    main()