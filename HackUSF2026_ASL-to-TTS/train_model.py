import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.neural_net import build_model, train

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__), "data", "gesture_landmarks.csv")
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "data", "asl_model.pth")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE    = 32    # Number of rows fed to the network per weight update
EPOCHS        = 50    # Full passes over the entire dataset
LEARNING_RATE = 0.001


# ── Dataset ───────────────────────────────────────────────────────────────────
# PyTorch's Dataset contract requires exactly two methods:
#   __len__  → total number of samples (so DataLoader knows when an epoch ends)
#   __getitem__ → returns one (X, y) pair by integer index
# DataLoader calls these under the hood to build batches automatically.
class ASLDataset(Dataset):

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        # ── Label encoding ────────────────────────────────────────────────────
        # Neural networks need integer targets, not strings. We build a sorted
        # mapping once here so the same encoding is used consistently:
        #   {'A': 0, 'B': 1, ..., 'Z': 25}
        # sorted() guarantees the mapping is deterministic across runs —
        # if the CSV row order changes, 'A' still maps to 0.
        unique_labels = sorted(df["label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes  = len(unique_labels)

        # ── Tensors ───────────────────────────────────────────────────────────
        # X: drop the label column, convert the remaining 63 float columns to a
        #    Float32 tensor. Float32 is PyTorch's default numeric type — using
        #    float64 would cause a type mismatch with nn.Linear weights.
        X = df.drop(columns=["label"]).values
        self.X = torch.tensor(X, dtype=torch.float32)

        # y: map each string label to its integer index, then store as Long
        #    (int64). CrossEntropyLoss requires Long targets specifically.
        y = df["label"].map(self.label_to_idx).values
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        # Called by DataLoader to know when one full epoch is complete.
        return len(self.X)

    def __getitem__(self, idx: int):
        # DataLoader calls this with a list of indices to build each batch.
        # Returning a tuple (features, label) is the standard convention that
        # the `for batch_X, batch_y in dataloader` loop in train() unpacks.
        return self.X[idx], self.y[idx]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Build dataset and wrap it in a DataLoader.
    # shuffle=True randomizes the row order each epoch so the model doesn't
    # memorize the sequence of batches — critical for generalization.
    dataset    = ASLDataset(CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")
    print(f"Label map: {dataset.label_to_idx}\n")

    # Initialize model, loss function, and optimizer from neural_net.py.
    # num_classes is read from the dataset so this script adapts automatically
    # if you add or remove gesture labels from the CSV.
    model, loss_fn, optimizer, device = build_model(
        num_classes=dataset.num_classes,
        learning_rate=LEARNING_RATE,
    )
    print(f"Training on: {device}")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}\n")

    # Run the training loop defined in neural_net.py.
    model = train(model, loss_fn, optimizer, dataloader, device, epochs=EPOCHS)

    # ── Save ──────────────────────────────────────────────────────────────────
    # torch.save persists two things needed to reconstruct the model later:
    #   state_dict   — all learned weight and bias tensors
    #   label_map    — the idx→label mapping so main.py can decode predictions
    #                  back to letters ('A', 'B', …) without re-reading the CSV
    torch.save(
        {
            "state_dict": model.state_dict(),
            "idx_to_label": dataset.idx_to_label,
            "num_classes": dataset.num_classes,
        },
        SAVE_PATH,
    )
    print(f"\nModel saved to {SAVE_PATH}")
