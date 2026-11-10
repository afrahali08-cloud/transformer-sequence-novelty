"""Train on five public UCI control-chart classes; hold out downward shifts."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

from model import CLASS_NAMES, SequenceEncoder, novelty_reference, novelty_score

DATA = Path(__file__).parent / "data" / "synthetic_control.data"


def split_indices(seed):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for category in range(5):
        ids = rng.permutation(np.arange(100 * category, 100 * (category + 1)))
        for result, portion in zip(parts, (ids[:60], ids[60:80], ids[80:])):
            result.extend(portion.tolist())
    return [np.array(part) for part in parts]


def auc(known_scores, unknown_scores):
    n, m = len(known_scores), len(unknown_scores)
    ranks = rankdata(np.r_[known_scores, unknown_scores])
    return float((ranks[n:].sum() - m * (m + 1) / 2) / (n * m))


def run(out, seed=7, epochs=100):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    raw = np.loadtxt(DATA, dtype=np.float32)
    if raw.shape != (600, 60) or not np.isfinite(raw).all():
        raise ValueError("Expected the original finite 600 x 60 UCI data")
    labels = torch.arange(600) // 100
    train, valid, test = split_indices(seed)
    mean, scale = float(raw[train].mean()), float(raw[train].std())
    x = torch.from_numpy((raw - mean) / scale)
    model = SequenceEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    criterion = nn.CrossEntropyLoss()
    history, best_loss, best_state, best_epoch = [], float("inf"), None, 0
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = torch.tensor(train)[torch.randperm(len(train))]
        loss_sum = 0.
        for ids in shuffled.split(32):
            optimizer.zero_grad()
            logits, _ = model(x[ids])
            loss = criterion(logits, labels[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x[valid])
            validation_loss = float(criterion(logits, labels[valid]))
        history.append((epoch, loss_sum / len(train), validation_loss))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    training_seconds = perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, train_embedding = model(x[train])
        centers, variance = novelty_reference(train_embedding, labels[train])
        _, validation_embedding = model(x[valid])
        threshold = float(torch.quantile(novelty_score(validation_embedding, centers, variance), .95))
        start = perf_counter()
        logits, embedding = model(x[test])
        _, unknown_embedding = model(x[500:])
        prediction = logits.argmax(1)
        known_scores = novelty_score(embedding, centers, variance).numpy()
        unknown_scores = novelty_score(unknown_embedding, centers, variance).numpy()
        inference_seconds = perf_counter() - start
    truth = labels[test].numpy()
    predictions = prediction.numpy()
    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (truth, predictions), 1)
    f1 = 2 * np.diag(confusion) / np.maximum(confusion.sum(0) + confusion.sum(1), 1)
    # A simple supervised baseline using exactly the same training and test examples.
    raw_centers = np.stack([x[train][labels[train] == k].numpy().mean(0) for k in range(5)])
    baseline = ((x[test].numpy()[:, None] - raw_centers[None])**2).mean(2).argmin(1)
    report = dict(
        seed=seed, epochs=epochs, best_epoch=best_epoch, parameters=sum(p.numel() for p in model.parameters()),
        split=dict(train=train.tolist(), validation=valid.tolist(), test=test.tolist(),
                   novelty_test=list(range(500, 600))),
        data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        classification=dict(accuracy=float((predictions == truth).mean()), macro_f1=float(f1.mean()),
                            confusion_matrix=confusion.tolist(),
                            nearest_centroid_accuracy=float((baseline == truth).mean())),
        novelty=dict(held_out_class="downward shift", auroc=auc(known_scores, unknown_scores),
                     threshold=threshold, known_test_false_positive_rate=float((known_scores > threshold).mean()),
                     unknown_test_recall=float((unknown_scores > threshold).mean())),
        training_seconds=training_seconds, inference_200_sequences_seconds=inference_seconds,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        normalization=dict(mean=mean, scale=scale))
    torch.save(dict(state_dict=best_state, mean=mean, scale=scale, centers=centers,
                    variance=variance, threshold=threshold, classes=CLASS_NAMES), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "validation_loss"])
        writer.writerows(history)
    with (out / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_row_zero_based", "true_class", "predicted_known_class", "novelty_score", "flagged_novel"])
        writer.writerows((int(i), int(y), int(p), float(s), bool(s > threshold))
                         for i, y, p, s in zip(test, truth, predictions, known_scores))
        writer.writerows((i, 5, "", float(s), bool(s > threshold))
                         for i, s in zip(range(500, 600), unknown_scores))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    history_array = np.array(history)
    axes[0].plot(history_array[:, 0], history_array[:, 1], label="Training")
    axes[0].plot(history_array[:, 0], history_array[:, 2], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training history")
    axes[0].legend()
    axes[1].imshow(confusion, cmap="Blues")
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, str(confusion[i, j]), ha="center", va="center",
                         color="white" if confusion[i, j] > 10 else "black")
    axes[1].set(xlabel="Predicted class (0–4)", ylabel="True class (0–4)", title="Known-class test")
    bins = np.histogram_bin_edges(np.r_[known_scores, unknown_scores], bins=20)
    axes[2].hist(known_scores, bins=bins, alpha=.6, label="Known test")
    axes[2].hist(unknown_scores, bins=bins, alpha=.6, label="Unseen downward shift")
    axes[2].axvline(threshold, color="black", linestyle="--", label="Validation threshold")
    axes[2].set(xlabel="Embedding distance", ylabel="Sequences", title="Novelty test")
    axes[2].legend(fontsize=8)
    fig.savefig(out / "evaluation.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "split"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("Epochs must be positive and seed nonnegative")
    run(args.out, args.seed, args.epochs)
