from dataset import SpecVocoderDataset, collate_fn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from utils import find_audio_files, pad_to_nearest_divisible
from sklearn.model_selection import train_test_split
from vocoders.wavenet import WaveNet
from sklearn.metrics import  accuracy_score
from torchaudio.functional import mu_law_encoding

## Define Dataset 
audio_files = find_audio_files('data/')
train_files, val_files = train_test_split(audio_files, test_size=0.1)

train_dataset = SpecVocoderDataset(training_files=train_files)
val_dataset = SpecVocoderDataset(training_files=val_files)

## Create Dataloader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)

## Define Model
model = WaveNet(in_channels=80, residual_channels=128, skip_channels=128, out_channels=256, residual_blocks=19)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


## train script 
def train(model, train_dataloader, optimizer, criterion):
    model.train()
    
    for i, (melspec, waveform) in enumerate(train_dataloader):
        if torch.isnan(melspec).any():
            print("Inputs contain NaN values.")

        optimizer.zero_grad()

        output = model(melspec)

        # Check for NaN in output
        if torch.isnan(output).any():
            print("Output contains NaN values.")
            continue

        preds = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)

        compressed_audio = mu_law_encoding(waveform, 255)
        padded_output = pad_to_nearest_divisible(compressed_audio, 275)

        padded_output = padded_output.view(-1).long()  # Flatten if necessary

        if (padded_output < 0).any() or (padded_output >= num_classes).any():
            print("Padded output contains invalid target values.")
            continue

        loss_value = criterion(output, padded_output)

        if loss_value is None or torch.isnan(loss_value):
            print("Loss computation returned NaN. Check output and target shapes.")
            continue

        loss_value.backward()

        # Check gradients for NaN
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("Gradient contains NaN values.")
                break
        else:
            optimizer.step()  # Only update if no NaN in gradients

## evaluation script 
def validate(model, val_dataloader, criterion):
    model.eval()

    val_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():  # Disable gradients for validation
        for i, (melspec, waveform) in enumerate(val_dataloader):
            
            # Forward pass through the model
            output = model(melspec)

            # Compute predictions
            preds = torch.argmax(torch.softmax(output, axis=1), axis=1).squeeze(0)

            # Mu-law encoding of waveform
            compressed_audio = mu_law_encoding(waveform, 255)

            print(compressed_audio.shape)

            # Ensure the output and target sizes match (pad if needed)
            padded_output = pad_to_nearest_divisible(compressed_audio, 275)

            # Compute loss
            loss_value = criterion(output, padded_output)
            val_loss += loss_value.item()

            # Compute accuracy
            accuracy = accuracy_score(padded_output.squeeze(0), preds)
            total_accuracy += accuracy

            if i % 10 == 0:
                print(f"Validation Step {i}, Loss: {loss_value.item()}, Accuracy: {accuracy}")

    avg_val_loss = val_loss / len(val_dataloader)
    avg_accuracy = total_accuracy / len(val_dataloader)

    print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_accuracy}")


## train

def full_training_loop(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training Phase
        print("Training Phase:")
        train(model, train_dataloader, optimizer, criterion)

        # Validation Phase
        print("Validation Phase:")
        validate(model, val_dataloader, criterion)

        print(f"Epoch {epoch+1} completed.\n")

num_epochs = 20
full_training_loop(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs)