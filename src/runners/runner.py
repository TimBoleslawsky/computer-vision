import os
import time

import torch
import torch.nn as nn
from torch import optim


class Runner:
    """General purpose runner class."""

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3,
                                                   gamma=0.5)  # halves the learning rate every 3 epochs.

    def run_training(self, train_loader, val_loader, save_path):
        # Training Loop
        best_val_acc = 0.0
        patience = 3
        epochs_no_improve = 0
        early_stop = False

        total_time = 0
        num_epochs = 5

        for epoch in range(num_epochs):
            if early_stop:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            start = time.time()

            train_acc, train_loss = self.train(train_loader)
            val_acc = self.evaluate(val_loader)

            self.scheduler.step()

            # Early Stopping Check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the results dir if it does not exist.
                torch.save(self.model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    early_stop = True

            end = time.time()
            epoch_time = end - start
            total_time += epoch_time

            print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Train Loss = {train_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}, Time = {epoch_time:.2f}s")

        avg_time = total_time / (num_epochs + 1)
        print(f"\nAverage training time per epoch: {avg_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

    def run_evaluation(self, test_loader, save_path):
        self.model.load_state_dict(torch.load(save_path))
        self.model.eval()

        test_acc = self.evaluate(test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")

    def train(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return accuracy, avg_loss

    def evaluate(self, loader):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return correct / total
