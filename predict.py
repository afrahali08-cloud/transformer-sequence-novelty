"""Classify rows of 60 numeric readings using the saved checkpoint."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import SequenceEncoder, novelty_score


def predict(values, checkpoint):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 60 or not np.isfinite(x).all():
        raise ValueError("Expected finite values with shape (number_of_sequences, 60)")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SequenceEncoder()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in torch.from_numpy((x - saved["mean"]) / saved["scale"]).split(128):
            logits, embedding = model(batch)
            scores = novelty_score(embedding, saved["centers"], saved["variance"])
            for category, score in zip(logits.argmax(1).tolist(), scores.tolist()):
                rows.append((saved["classes"][category], score, score > saved["threshold"]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Whitespace-delimited rows, 60 values per row")
    parser.add_argument("--checkpoint", type=Path, default=Path("benchmark/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    results = predict(np.loadtxt(args.input, ndmin=2), args.checkpoint)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["nearest_known_class", "novelty_score", "flagged_novel"])
        writer.writerows(results)
    print(f"Wrote {len(results)} predictions to {args.out}")
