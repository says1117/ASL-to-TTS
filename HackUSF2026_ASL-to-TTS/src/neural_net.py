import os
import torch
import torch.nn as nn

# Base class for every neural network
# Inheriting from this class gives us access to automatic parameter tracking
# .to(device) method to move model to GPU if available
# And ability to call a model like a function (model(X) calls forward(X))
class ASLNeuralNetwork(nn.Module):

    def __init__ (self, num_classes: int):
        #always call parent __init__ first to set up the internal machinery that PyTorch needs to track parameters and do backpropagation
    #bookskeeps and manages layers automatically
        super().__init__()

        # nn.Sequential packages layers into an ordered pipeline. When we call
        # self.network(X), PyTorch feeds X through each layer left to right
        self.network = nn.Sequential(
            #256 neurons. fully-connected (dense) layer.
            # input is 63 because 21*3, output is 256 because its our bias
            # weight matrix
            nn.Linear(63, 256),  # Input layer: 63 features (21 joints * 3 coordinates)

            #Rectified Linear Unit: f(x) = max(0, x).
            # without non linearity between linear layers, stacking collapses them into a single linear transformation
            # ReLU breaks linearity and allows the network to learn complex, non-linear relationships in the data
            nn.ReLU(),
            
            nn.Linear(256, 128),

            
            nn.ReLU(),

            # Dropout: during each forward pass in training, randomly zeros
            # p=20% of neurons. This forces the network to learn redundant
            # representations — no single neuron can be relied upon — which
            # is the main mechanism that prevents overfitting. Dropout is
            # automatically disabled during model.eval() (inference).
            nn.Dropout(p=0.2),

            # ── Layer 3: 128 → 64 neurons ────────────────────────────────────
            nn.Linear(128, 64),
            nn.ReLU(),

            # ── Output layer: 64 → num_classes ───────────────────────────────
            # num_classes is passed in at construction time so the architecture
            # adapts automatically whether your dataset has 24 or 26 labels.
            # No activation here — CrossEntropyLoss expects raw logits (scores
            # before softmax). Applying softmax manually would double-apply it
            # since CrossEntropyLoss calls log_softmax internally.
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward() defines the computation graph for one pass of data through
        # the model. PyTorch traces this graph during .backward() to calculate
        # gradients via the chain rule.
        return self.network(x)

    def train_model(self, epochs: int = 20, lr: float = 0.001, batch_size: int = 32):
        import pandas as pd
        from torch.utils.data import TensorDataset, DataLoader

        csv_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'data', 'asl_landmarks_kaggle.csv')

        import numpy as np

        df = pd.read_csv(csv_path)
        # Remove del and space — we only classify A-Z
        df = df[df['label'].str.len() == 1]
        X = df.drop(columns=['label']).values.astype('float32')

        # Normalize every sample so the wrist is pinned to (0,0,0).
        # This matches the live normalization in tracker.normalize_hand()
        # so training and inference see the same coordinate space.
        X_3d  = X.reshape(-1, 21, 3)                        # (n, 21, 3)
        wrists = X_3d[:, 0, :]                               # (n, 3)  — wrist per sample
        X_3d  = X_3d - np.expand_dims(wrists, axis=1)       # broadcast across 21 joints
        X     = X_3d.reshape(-1, 63)                         # back to (n, 63)

        # Map string labels like 'A','B'... to integers 0,1,...
        labels = sorted(df['label'].unique())
        self._label_to_idx = {label: i for i, label in enumerate(labels)}
        self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}

        y = df['label'].map(self._label_to_idx).values.astype('int64')

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        train(self, loss_fn, optimizer, dataloader, self._device, epochs=epochs)

    def predict(self, landmarks: list, confidence_threshold: float = 0.70) -> str | None:
        if not hasattr(self, '_idx_to_label'):
            return None

        self.eval()
        device = getattr(self, '_device', torch.device('cpu'))
        X = torch.tensor([landmarks], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self(X)
            # Convert raw logits to probabilities. The highest probability
            # tells us how confident the model is in its top prediction.
            probs = torch.softmax(logits, dim=1)
            confidence, idx = probs.max(dim=1)

        # If the model isn't sure enough, return None so main.py shows nothing
        # rather than flashing a wrong letter.
        if confidence.item() < confidence_threshold:
            return None

        return self._idx_to_label.get(idx.item())


# ── Setup ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, learning_rate: float = 0.001):
    """
    Instantiate the model, loss function, and optimizer.
    Returns all three so train_model() can use them independently.
    """

    # Move computation to GPU if available, otherwise CPU.
    # .to(device) must be called before passing the model to the optimizer,
    # otherwise optimizer tensors live on a different device than model weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLNeuralNetwork(num_classes).to(device)

    # CrossEntropyLoss = log_softmax + NLLLoss in one step.
    # It expects raw logits from the model and integer class indices as targets
    # (not one-hot vectors). Internally it penalises the model proportionally
    # to how wrong and how confident it was on the wrong class.
    loss_fn = nn.CrossEntropyLoss()

    # Adam (Adaptive Moment Estimation) maintains a per-parameter learning rate
    # by tracking the first moment (mean of gradients) and second moment
    # (variance of gradients). In practice it converges faster than plain SGD
    # and is robust to choice of learning rate. lr=0.001 is the standard default.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer, device


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, loss_fn, optimizer, dataloader, device, epochs: int = 20):
    """
    Train `model` for `epochs` full passes over `dataloader`.
    Call build_model() first to get model / loss_fn / optimizer / device.
    """

    # model.train() switches layers that behave differently during training vs
    # inference — specifically Dropout (active) and BatchNorm (uses batch stats).
    # Forgetting this after a model.eval() call is a common silent bug.
    model.train()

    for epoch in range(1, epochs + 1):

        running_loss = 0.0  # Accumulates loss across all batches this epoch

        for batch_X, batch_y in dataloader:

            # Move this batch's tensors to the same device as the model.
            # If model is on GPU and tensors are on CPU, PyTorch will error.
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # ── Step 1: Zero the gradients ────────────────────────────────────
            # PyTorch ACCUMULATES gradients into .grad by default (useful for
            # some advanced techniques like gradient accumulation). For standard
            # training we want fresh gradients each batch, so we clear them
            # manually. Forgetting this is one of the most common PyTorch bugs —
            # gradients from the previous batch would add to the current batch,
            # producing incorrect weight updates.
            optimizer.zero_grad()

            # ── Step 2: Forward pass ──────────────────────────────────────────
            # Pass the batch through the network. PyTorch builds a computation
            # graph (a DAG of operations) as it goes, recording every
            # multiplication and addition. This graph is what .backward() walks.
            predictions = model(batch_X)        # shape: (batch_size, num_classes)

            # ── Step 3: Compute loss ──────────────────────────────────────────
            # Measures how wrong the predictions are. Lower = better.
            # predictions: raw logits  |  batch_y: integer class indices
            loss = loss_fn(predictions, batch_y)

            # ── Step 4: Backward pass ─────────────────────────────────────────
            # .backward() walks the computation graph backwards from the scalar
            # loss value, applying the chain rule at every node to compute
            # dLoss/dWeight for every parameter in the network. The result is
            # stored in each parameter's .grad attribute, ready for the optimizer.
            # Mathematically: if loss = f(g(h(x))), backward computes
            #   dL/dW = dL/df · df/dg · dg/dh · dh/dW   (chain rule)
            loss.backward()

            # ── Step 5: Update weights ────────────────────────────────────────
            # Adam reads each parameter's .grad, adjusts the learning rate per
            # parameter using its moment estimates, and subtracts the scaled
            # gradient from the weight:  W = W - lr * adjusted_gradient
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch:>3}/{epochs}  |  avg loss: {avg_loss:.4f}")

    print("Training complete.")
    return model
