import sys
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from snn_model import PathSNN, CLASS_NAMES


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset_snn"
IMAGE_DIR = DATASET_DIR / "images"
CSV_PATH = DATASET_DIR / "labels.csv"

MODEL_PATH = REPO_ROOT / "models" / "path_snn.pth"

IMAGE_SIZE = 32
NUM_STEPS = 20

HIDDEN_SIZE = 128
BETA = 0.9

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

VAL_SPLIT = 0.20

SEED = 42


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class PathDataset(Dataset):
    def __init__(self, rows, image_dir):
        self.rows = rows
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        filename, label = self.rows[idx]

        path = self.image_dir / filename

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Immagine non leggibile: {path}")

        # OpenCV BGR -> RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # [0,255] -> [0,1], required by rate coding.
        image = image.astype(np.float32) / 255.0

        # [H,W,C] -> [C,H,W] -> flatten.
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.reshape(-1)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def load_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {CSV_PATH}\n"
            "Esegui prima collect_snn_dataset.py."
        )

    rows = []

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row["filename"]
            label = int(row["label"])

            path = IMAGE_DIR / filename

            if path.exists():
                rows.append((filename, label))

    if len(rows) < 20:
        raise RuntimeError(
            f"Dataset troppo piccolo: {len(rows)} immagini."
        )

    return rows


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def accuracy_from_membrane(mem_rec, targets):
    """
    Uses the final membrane potential as the classification score.
    """
    scores = mem_rec[-1]
    predicted = scores.argmax(dim=1)

    correct = (predicted == targets).sum().item()
    return correct, targets.numel()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # SNN forward through all timesteps.
        _, mem_rec = model(
            data,
            num_steps=NUM_STEPS,
        )

        # PPT: sum loss for every timestep.
        loss = torch.zeros((), device=device)

        for step in range(NUM_STEPS):
            loss = loss + criterion(
                mem_rec[step],
                targets,
            )

        loss = loss / NUM_STEPS

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)

        correct, count = accuracy_from_membrane(
            mem_rec,
            targets,
        )
        total_correct += correct
        total_items += count

    return (
        total_loss / total_items,
        total_correct / total_items,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)

        _, mem_rec = model(
            data,
            num_steps=NUM_STEPS,
        )

        loss = torch.zeros((), device=device)

        for step in range(NUM_STEPS):
            loss = loss + criterion(
                mem_rec[step],
                targets,
            )

        loss = loss / NUM_STEPS

        total_loss += loss.item() * targets.size(0)

        correct, count = accuracy_from_membrane(
            mem_rec,
            targets,
        )
        total_correct += correct
        total_items += count

    return (
        total_loss / total_items,
        total_correct / total_items,
    )


def make_loaders(rows):
    random.shuffle(rows)

    split = int(len(rows) * (1.0 - VAL_SPLIT))

    train_rows = rows[:split]
    val_rows = rows[split:]

    if len(val_rows) == 0:
        raise RuntimeError("Validation set vuoto.")

    train_dataset = PathDataset(
        train_rows,
        IMAGE_DIR,
    )

    val_dataset = PathDataset(
        val_rows,
        IMAGE_DIR,
    )

    # Weighted sampler to reduce class imbalance.
    labels = [label for _, label in train_rows]

    class_counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    class_weights = np.zeros_like(
        class_counts,
        dtype=np.float64,
    )

    for i, count in enumerate(class_counts):
        if count > 0:
            class_weights[i] = 1.0 / count

    sample_weights = [
        class_weights[label]
        for label in labels
    ]

    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Class counts:", class_counts.tolist())

    return train_loader, val_loader


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("TRAINING SNN - RED PATH FOLLOWING")
    print("=" * 60)
    print("Device:", device)
    print("Image:", f"{IMAGE_SIZE}x{IMAGE_SIZE} RGB")
    print("Input features:", IMAGE_SIZE * IMAGE_SIZE * 3)
    print("Hidden:", HIDDEN_SIZE)
    print("Outputs:", len(CLASS_NAMES))
    print("Timesteps:", NUM_STEPS)
    print("Beta:", BETA)
    print()

    rows = load_rows()
    train_loader, val_loader = make_loaders(rows)

    model = PathSNN(
        input_size=IMAGE_SIZE * IMAGE_SIZE * 3,
        hidden_size=HIDDEN_SIZE,
        num_outputs=len(CLASS_NAMES),
        beta=BETA,
    ).to(device)

    # CrossEntropy on membrane potentials.
    # Surrogate gradient is used internally by the LIF neurons.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f} "
            f"acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": IMAGE_SIZE * IMAGE_SIZE * 3,
                    "hidden_size": HIDDEN_SIZE,
                    "num_outputs": len(CLASS_NAMES),
                    "beta": BETA,
                    "image_size": IMAGE_SIZE,
                    "num_steps": NUM_STEPS,
                    "class_names": CLASS_NAMES,
                },
                MODEL_PATH,
            )

            print(
                f"  -> modello migliore salvato: "
                f"{MODEL_PATH}"
            )

    print()
    print("Training completato.")
    print(
        f"Migliore validation accuracy: "
        f"{best_val_acc*100:.2f}%"
    )


if __name__ == "__main__":
    main()
