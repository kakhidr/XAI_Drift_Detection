import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class IDS_MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_mlp(cfg, input_dim: int) -> IDS_MLP:
    mcfg = cfg["model"]
    return IDS_MLP(
        input_dim=input_dim,
        hidden_dims=mcfg.get("hidden_dims", [128, 64]),
        num_classes=mcfg.get("num_classes", 2),
        dropout=mcfg.get("dropout", 0.1),
    )


def train_model(model, X_train, y_train, cfg, progress_cb=None):
    """
    Train the model with early stopping and LR scheduling.

    Parameters
    ----------
    progress_cb : callable(epoch, epochs, train_loss, val_loss), optional
        Called after each epoch for UI updates.

    Returns
    -------
    model, history : trained model and list of per-epoch dicts
    """
    mcfg = cfg["model"]
    epochs = mcfg.get("epochs", 20)
    batch_size = mcfg.get("batch_size", 256)
    lr = mcfg.get("lr", 1e-3)
    patience = mcfg.get("patience", 5)
    seed = cfg["run"]["seed"]

    device = torch.device("cuda" if cfg["run"].get("use_cuda", False) and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Split off 20% for validation
    from sklearn.model_selection import train_test_split as _split

    X_tr, X_val, y_tr, y_val = _split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    early_stop_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if progress_cb:
            progress_cb(epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history