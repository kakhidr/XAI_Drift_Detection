def build_mlp(cfg, input_dim: int):
    """Return a torch.nn.Module that outputs logits [N, num_classes]."""
    raise NotImplementedError("Implement MLP builder.")

def train_model(model, X_train, y_train, cfg):
    """Train model and return trained model. Use cfg.model epochs/batch_size/lr."""
    raise NotImplementedError("Implement model training.")