# test_run.py
# Copyright 2022 DeepMind Technologies Limited.
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
"""Unit tests for the run.py module of clrs_pytorch."""

import os
import json
import tempfile
import unittest
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from absl import app, flags, logging

import clrs_pytorch
from clrs_pytorch._src import specs, losses, samplers, decoders
import run 

_Location = specs.Location

# ---------------------------------------------------------------------------
# Dummy Classes for Testing Feedback and Model
# ---------------------------------------------------------------------------
class DummyDataPoint:
    """A minimal dummy data point mimicking expected feedback objects."""
    def __init__(self, name, data, location, type_):
        self.name = name
        self.data = data
        self.location = location
        self.type_ = type_

class DummyFeatures:
    """Dummy features container with inputs, hints, and lengths."""
    def __init__(self, batch_size, num_nodes, max_timesteps):
        # Dummy inputs: shape [batch_size, num_nodes, feature_dim]
        self.inputs = [DummyDataPoint('dummy_input', torch.ones(batch_size, num_nodes), specs.Location.NODE, specs.Type.SCALAR)]
        # Dummy hints: shape [batch_size, feature_dim]
        self.hints = [DummyDataPoint('dummy_hint', torch.ones(max_timesteps, batch_size,  num_nodes), specs.Location.NODE, specs.Type.SCALAR)]
        # Dummy lengths: one integer per sample.
        self.lengths = torch.ones(batch_size, dtype=torch.int32)

class DummyFeedback:
    """A dummy feedback object containing features and outputs."""
    def __init__(self, batch_size=32, num_nodes=4, max_timesteps = 4):
        self.features = DummyFeatures(batch_size, num_nodes, max_timesteps)
        self.outputs = [DummyDataPoint('dummy_output', torch.ones(batch_size, num_nodes),  specs.Location.NODE, specs.Type.SCALAR)]

class DummyModel(nn.Module):
    """A simple dummy model for testing training functions."""
    def __init__(self, input_dim , output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, feedback, algo_idx, repred=False, return_hints=False, return_all_outputs=False):
        batch_size = feedback.outputs[0].data.shape[0]
        num_nodes = feedback.outputs[0].data.shape[1]
        # Constant output prediction.
        output_preds = {'dummy_output': self.linear(feedback.features.inputs[0].data)}
        # Return hint predictions as a list of dictionaries.
        hint_preds = [{'dummy_hint': torch.ones( batch_size, num_nodes)}, 
                      {'dummy_hint': torch.ones( batch_size, num_nodes)},
                      {'dummy_hint': torch.ones( batch_size, num_nodes)}]
        return output_preds, hint_preds

# Monkey-patch decoders.postprocess to be identity for testing predict().
decoders.postprocess = lambda spec, outs, sinkhorn_temperature, sinkhorn_steps, hard: outs

# ---------------------------------------------------------------------------
# Unit Tests for run.py Functions
# ---------------------------------------------------------------------------
class RunModuleTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 32
        self.feature_dim = 4
        self.num_nodes = 4
        self.max_timesteps = 4
        self.dummy_feedback = DummyFeedback(batch_size=self.batch_size,
                                            num_nodes=self.num_nodes,
                                            max_timesteps=self.max_timesteps)
        self.model = DummyModel(input_dim=self.num_nodes, output_dim=self.feature_dim).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible random numbers."""
        run.set_seed(123)
        a = np.random.rand(5)
        run.set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_move_feedback_to_device(self):
        """Test that move_feedback_to_device converts all data to torch.Tensors on the target device."""
        np_data = np.ones((self.batch_size, self.feature_dim))
        self.dummy_feedback.outputs[0].data = np_data
        moved = run.move_feedback_to_device(self.dummy_feedback, self.device)
        self.assertIsInstance(moved.outputs[0].data, torch.Tensor)
        self.assertEqual(moved.outputs[0].data.device.type, self.device.type)

    def test_loss_function_returns_scalar(self):
        """Test that loss() returns a scalar tensor."""
        output_preds = {'dummy_output': torch.ones(self.batch_size, self.num_nodes)}
        hint_preds = [{'dummy_hint': torch.ones(self.batch_size, self.num_nodes)},
                      {'dummy_hint': torch.ones( self.batch_size, self.num_nodes)},
                      {'dummy_hint': torch.ones(self.batch_size, self.num_nodes)}]
        loss_val = run.loss(self.dummy_feedback, output_preds, hint_preds, self.device, decode_hints=True)
        self.assertIsInstance(loss_val, torch.Tensor)
        self.assertEqual(loss_val.dim(), 0)

    def test_train_function_updates_model(self):
        """Test that a training step returns a loss and updates model parameters."""
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        lss = run.train(self.model, self.optimizer, self.dummy_feedback, 0, grad_clip_max_norm=1.0, device=self.device)
        self.assertGreater(lss.item(), 0)
        for k, v in self.model.state_dict().items():
            self.assertFalse(torch.equal(v, initial_state[k]))

    def test_save_and_restore_model(self):
        """Test that saving and restoring the model preserves its state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
            run.save_model(self.model, checkpoint_path)
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            for param in self.model.parameters():
                param.data.add_(1.0)
            run.restore_model(self.model, checkpoint_path)
            restored_state = self.model.state_dict()
            for k, v in original_state.items():
                self.assertTrue(torch.allclose(v, restored_state[k]))

if __name__ == '__main__':
    unittest.main()
