import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import SequenceEncoder, novelty_score
from predict import predict
from train import auc, split_indices


class TransformerTests(unittest.TestCase):
    def test_forward_gradients_and_distinct_layers(self):
        torch.manual_seed(4)
        model = SequenceEncoder()
        logits, embedding = model(torch.randn(3, 60))
        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertEqual(tuple(embedding.shape), (3, 32))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.patch.weight.grad).all())
        layers = model.encoder.layers
        self.assertFalse(torch.equal(layers[0].linear1.weight, layers[1].linear1.weight))

    def test_split_excludes_unknown_and_has_no_overlap(self):
        parts = split_indices(7)
        self.assertEqual([len(p) for p in parts], [300, 100, 100])
        self.assertEqual(len(set(np.concatenate(parts))), 500)
        self.assertLess(max(np.concatenate(parts)), 500)

    def test_distance_and_auc(self):
        centers = torch.zeros(5, 32)
        score = novelty_score(torch.stack([torch.zeros(32), torch.ones(32)]),
                              centers, torch.ones(32))
        torch.testing.assert_close(score, torch.tensor([0., 1.]))
        self.assertEqual(auc([0, 1], [2, 3]), 1.)
        self.assertEqual(auc([1, 1], [1, 1]), .5)

    def test_checkpoint_round_trip_prediction(self):
        torch.manual_seed(2)
        model = SequenceEncoder().eval()
        x = torch.randn(2, 60)
        with torch.no_grad():
            expected = model(x)[0].argmax(1).tolist()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(dict(state_dict=model.state_dict(), mean=0., scale=1.,
                            centers=torch.zeros(5, 32), variance=torch.ones(32),
                            threshold=100., classes=list("abcde")), path)
            result = predict(x.numpy(), path)
        self.assertEqual([row[0] for row in result], [list("abcde")[i] for i in expected])

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            SequenceEncoder()(torch.ones(2, 59))
        with self.assertRaises(ValueError):
            predict(np.full((1, 60), np.nan), Path("unused"))
