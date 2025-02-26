# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for loss utility functions in losses.py."""

import unittest
import copy
import torch
import numpy as np

from clrs_pytorch._src import losses, specs, probing

# -----------------------------------------------------------------------------
# Dummy DataPoint for Testing
# -----------------------------------------------------------------------------

class DummyDataPoint:
    """A simple dummy data point mimicking probing.DataPoint."""
    def __init__(self, name, location, type_, data):
        self.name = name
        self.location = location
        self.type_ = type_
        self.data = data

# -----------------------------------------------------------------------------
# Losses Test Suite
# -----------------------------------------------------------------------------

class LossesTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.nb_nodes = 5
        self.EPS = 1e-12
        # For testing, we assume that these constants are defined in specs.
        self._Type = specs.Type
        self._OutputClass = specs.OutputClass

    def test_output_loss_scalar(self):
        """Test output_loss for a scalar truth."""
        truth_data = torch.tensor([1.0, 2.0, 3.0])
        dp = DummyDataPoint("scalar", "node", self._Type.SCALAR, truth_data)
        pred = torch.tensor([2.0, 2.0, 2.0])
        # Expected loss = mean((pred - truth)^2) = mean([1, 0, 1]) = 2/3.
        expected_loss = (1.0 + 0.0 + 1.0) / 3.0
        loss_val = losses.output_loss(dp, pred, self.nb_nodes, self.device)
        self.assertAlmostEqual(loss_val.item(), expected_loss, places=4)

    def test_output_loss_mask(self):
        """Test output_loss for a MASK truth.
        
        Here we assume that specs.OutputClass.MASKED equals 0.
        """
        masked_value = self._OutputClass.MASKED  # Assumed to be -1.
        truth_data = torch.tensor([1.0, masked_value, 1.0])
        dp = DummyDataPoint("mask", "node", self._Type.MASK, truth_data)
        pred = torch.tensor([2.0, 2.0, 2.0])
        # For unmasked elements (indices 0 and 2):
        #   loss = max(2, 0) - 2*1 + log1p(exp(-|2|)) = 2 - 2 + log1p(exp(-2))
        expected_elem = float(torch.log1p(torch.exp(torch.tensor(-2.0))))
        expected_loss = (expected_elem + expected_elem) / 2.0
        loss_val = losses.output_loss(dp, pred, self.nb_nodes, self.device)
        self.assertAlmostEqual(loss_val.item(), expected_loss, places=4)

    def test_output_loss_mask_one(self):
        """Test output_loss for a MASK_ONE (or CATEGORICAL) truth."""
        # For this branch, assume:
        #   - specs.OutputClass.MASKED equals -1.
        #   - specs.OutputClass.POSITIVE equals 1.
        truth_data = torch.tensor([1.0, 0.0, 0.0])
        dp = DummyDataPoint("mask_one", "node", self._Type.MASK_ONE, truth_data)
        pred = torch.tensor([1.0, 2.0, 3.0])
        logsm = torch.nn.functional.log_softmax(pred, dim=-1)
        numerator = -torch.sum(truth_data * (truth_data != 0).float() * logsm)
        denominator = torch.sum(truth_data == 1).float()
        expected_loss = numerator / denominator
        loss_val = losses.output_loss(dp, pred, self.nb_nodes, self.device)
        self.assertAlmostEqual(loss_val.item(), expected_loss.item(), places=4)

    def test_output_loss_pointer(self):
        """Test output_loss for a POINTER truth."""
        # For POINTER, truth.data is an integer tensor.
        truth_data = torch.tensor([2])
        dp = DummyDataPoint("pointer", "node", self._Type.POINTER, truth_data)
        # pred should have shape [1, nb_nodes]. Using zeros so that log_softmax is uniform.
        pred = torch.zeros(1, self.nb_nodes)
        expected_loss = torch.log(torch.tensor(self.nb_nodes, dtype=torch.float32))
        loss_val = losses.output_loss(dp, pred, self.nb_nodes, self.device)
        self.assertAlmostEqual(loss_val.item(), expected_loss.item(), places=4)

    def test_output_loss_permutation_pointer(self):
        """Test output_loss for a PERMUTATION_POINTER truth."""
        truth_data = torch.tensor([[0.0, 1.0],
                                   [1.0, 0.0]])
        dp = DummyDataPoint("perm_pointer", "node", self._Type.PERMUTATION_POINTER, truth_data)
        pred = torch.tensor([[0.1, 0.2],
                             [0.3, 0.4]])
        # Row-wise: row0: - (0*0.1 + 1*0.2) = -0.2; row1: - (1*0.3 + 0*0.4) = -0.3.
        expected_loss = (-0.2 + -0.3) / 2.0
        loss_val = losses.output_loss(dp, pred, self.nb_nodes, self.device)
        self.assertAlmostEqual(loss_val.item(), expected_loss, places=4)

    def test_hint_loss(self):
        """Test hint_loss for a scalar truth with time-dimension data."""
        # Create dummy truth data with a time dimension.
        # Use a 2D tensor so that time dimension and batch dimension are explicit.
        truth_data = torch.tensor([[1.0], [2.0], [3.0]])  # Shape: [3, 1]
        dp = DummyDataPoint("hint", "node", self._Type.SCALAR, truth_data)
        # Provide preds as a list of column vectors, but squeeze the last dimension.
        preds = [torch.tensor([[3.0]]).squeeze(-1), torch.tensor([[4.0]]).squeeze(-1)]
        # After squeezing, each prediction has shape [1], so stacking yields shape [2].
        # Expected: loss = (pred - truth[1:])^2 = ([3-2, 4-3])^2 = ([1, 1]).
        # Average loss = 1.
        lengths = torch.tensor([3])
        loss_val = losses.hint_loss(dp, preds, lengths, self.nb_nodes, self.device)
        self.assertAlmostEqual(loss_val.item(), 1.0, places=4)

    def test_is_not_done_broadcast(self):
        """Test that _is_not_done_broadcast returns the correct mask."""
        lengths = [3, 4]
        i = 1
        tensor = torch.ones(2, 5)
        mask = losses._is_not_done_broadcast(lengths, i, tensor, self.device)
        expected = torch.ones(2, 5)
        self.assertTrue(torch.allclose(mask, expected, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
