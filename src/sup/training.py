from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .models import DeepNeuralNetwork


def train_dnn_with_early_stopping(
    X_train, y_train, X_val, y_val, input_dim,
    max_epochs=40, batch_size=64, lr=0.001, patience=5,
    save_path="outputs/supervised/models/best_dnn_model.pth",
):
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepNeuralNetwork(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_losses": [], "val_losses": [], "train_accuracies": [], "val_accuracies": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = running_loss / len(train_ds)
        train_acc = correct / total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss_sum += loss.item() * batch_X.size(0)
                val_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss = val_loss_sum / len(val_ds)
        val_acc = val_correct / val_total

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accuracies"].append(train_acc)
        history["val_accuracies"].append(val_acc)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"--> Early stopping triggered at epoch {epoch} !!!!!!")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, history
