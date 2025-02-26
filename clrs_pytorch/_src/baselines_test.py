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
"""Unit tests for the PyTorch CLRS BaselineModel."""

from dataclasses import dataclass, replace
from typing import List, Union
import copy
import unittest

from absl.testing import absltest, parameterized
import numpy as np
import torch

from clrs_pytorch._src import baselines, nets, processors, samplers, specs

# -----------------------------------------------------------------------------
# Dummy Data Structures for Testing
# -----------------------------------------------------------------------------

@dataclass
class DummyField:
    name: str
    data: torch.Tensor
    location: Union[str, None] = None
    type_: Union[str, None] = None

@dataclass
class DummyFeatures:
    inputs: List[DummyField]
    hints: List[DummyField]

@dataclass
class DummyFeedback:
    features: DummyFeatures
    outputs: List[DummyField]

    def _replace(self, **kwargs):
        return replace(self, **kwargs)

def create_dummy_feedback(batch_size: int = 4,
                          input_dim: int = 10,
                          hint_dim: int = 5,
                          output_dim: int = 10) -> DummyFeedback:
    """Creates a minimal dummy feedback for testing."""
    # Create dummy fields.
    input_field = DummyField(name="input", data=torch.randn(batch_size, input_dim))
    hint_field = DummyField(name="hint", data=torch.randn(batch_size, hint_dim))
    # For outputs, use pointer type at NODE location.
    output_field = DummyField(name="output",
                              data=torch.randn(batch_size, output_dim),
                              location=specs.Location.NODE,
                              type_=specs.Type.POINTER)
    features = DummyFeatures(inputs=[input_field], hints=[hint_field])
    return DummyFeedback(features=features, outputs=[output_field])

# -----------------------------------------------------------------------------
# Dummy Network for Monkey-Patching
# -----------------------------------------------------------------------------

import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize with random values instead of zeros.
        self.dummy_param = nn.Parameter(torch.randn(1, 10))
    
    def forward(self, features, repred, algorithm_index, return_hints, return_all_outputs):
        batch_size = features[0].inputs[0].data.shape[0]
        dummy_output = self.dummy_param.expand(batch_size, -1)
        dummy_hint = torch.zeros(batch_size, 5)
        return dummy_output, dummy_hint

# Monkey-patch nets.Net with DummyNet for testing.
nets.Net = DummyNet

# -----------------------------------------------------------------------------
# Dummy Spec for Testing
# -----------------------------------------------------------------------------

def create_dummy_spec() -> dict:
    """Creates a minimal dummy spec dictionary.

    The spec is not used directly in these tests (beyond being passed to the model),
    so we provide minimal values.
    """
    # For simplicity, we use a dict with a single key.
    # The tuple values are (some identifier, location, type).
    return {"dummy": (None, specs.Location.NODE, specs.Type.POINTER)}

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

class BaselineModelTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cpu")
        self.dummy_feedback = create_dummy_feedback()
        self.dummy_spec = create_dummy_spec()

        # Use a simple processor factory.
        self.processor_factory = processors.get_processor_factory('mpnn', use_ln=True, nb_triplet_fts=0)
        self.common_args = dict(
            processor_factory=self.processor_factory,
            hidden_dim=8,
            learning_rate=0.01,
            checkpoint_path='/tmp/clrs3',
            freeze_processor=False,
            dropout_prob=0.0,
            hint_teacher_forcing=0.0,
            hint_repred_mode='soft',
            nb_msg_passing_steps=1,
        )

    def test_invalid_hint_configuration(self):
        """Test that instantiating with encode_hints True and decode_hints False raises ValueError."""
        with self.assertRaises(ValueError):
            baselines.BaselineModel(
                spec=self.dummy_spec,
                dummy_trajectory=[self.dummy_feedback],
                device=self.device,
                encode_hints=True,
                decode_hints=False,
                **self.common_args
            )

    def test_forward_pass_single_algorithm(self):
        """Test that the forward pass returns a tuple of outputs for a single algorithm."""
        model = baselines.BaselineModel(
            spec=self.dummy_spec,
            dummy_trajectory=[self.dummy_feedback],
            device=self.device,
            decode_hints=True,
            encode_hints=True,
            **self.common_args
        )
        model.to(self.device)
        output_preds, hint_preds = model(
            self.dummy_feedback,
            algorithm_index=0,
            repred=False,
            return_hints=True,
            return_all_outputs=False
        )
        # Check that outputs are tensors of the expected shapes.
        self.assertIsInstance(output_preds, torch.Tensor)
        self.assertIsInstance(hint_preds, torch.Tensor)
        batch_size = self.dummy_feedback.features.inputs[0].data.shape[0]
        self.assertEqual(list(output_preds.shape), [batch_size, 10])
        self.assertEqual(list(hint_preds.shape), [batch_size, 5])

    def test_training_step_updates_parameters(self):
        """Test that a training step produces nonzero parameter updates."""
        model = baselines.BaselineModel(
            spec=self.dummy_spec,
            dummy_trajectory=[self.dummy_feedback],
            device=self.device,
            decode_hints=True,
            encode_hints=True,
            **self.common_args
        )
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Save the initial parameters.
        init_state = copy.deepcopy(model.state_dict())

        # Perform a forward pass, compute a dummy loss, and update.
        output_preds, hint_preds = model(
            self.dummy_feedback,
            algorithm_index=0,
            repred=False,
            return_hints=True,
            return_all_outputs=False
        )
        # Define a dummy loss (mean squared error on outputs).
        loss = torch.mean(output_preds ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        new_state = model.state_dict()
        # Compare the parameters to ensure they have been updated.
        total_change = 0.0
        for key in init_state:
            diff = torch.sum(torch.abs(init_state[key] - new_state[key])).item()
            total_change += diff
        self.assertGreater(total_change, 1e-6)

if __name__ == '__main__':
    absltest.main()
