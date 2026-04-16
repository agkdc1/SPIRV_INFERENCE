#!/usr/bin/env python3
"""PyTorch reference: same MLP architecture + hyperparams for SPIR-V comparison."""
import json, time, os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

BATCH = 64
EPOCHS = 5
LR = 0.001
SEED = 42

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pytorch] device: {device}")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tx)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tx)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                                drop_last=True, generator=torch.Generator().manual_seed(SEED))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH, shuffle=False, drop_last=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # Kaiming init (same as SPIR-V version)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    results = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            if batch_idx % 100 == 0:
                print(f"  epoch {epoch}/{EPOCHS} batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = total_loss / count

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        acc = correct / total
        print(f"=== Epoch {epoch}/{EPOCHS}: loss={avg_loss:.4f}, test_acc={acc*100:.2f}% ===")
        results.append({"epoch": epoch, "train_loss": avg_loss, "test_accuracy": acc})

    elapsed = time.time() - start
    final = results[-1]
    print(f"\n[pytorch] Final accuracy: {final['test_accuracy']*100:.2f}%")
    print(f"[pytorch] Total time: {elapsed:.1f}s")
    print(f"[pytorch] Device: {device}")

    report = {
        "framework": "PyTorch",
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "final_test_accuracy": final["test_accuracy"],
        "final_train_loss": final["train_loss"],
        "total_time_secs": elapsed,
        "epoch_results": results,
    }
    out = os.path.join(os.path.dirname(__file__), "PYTORCH_RESULT.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {out}")

if __name__ == "__main__":
    main()
