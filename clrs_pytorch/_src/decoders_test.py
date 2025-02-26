# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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

"""Unit tests for `decoders.py`."""

import unittest
import torch
import torch.nn.functional as F

from clrs_pytorch._src import decoders


class DecodersTest(unittest.TestCase):
    def test_log_sinkhorn(self):
        # Create a random tensor of shape [10, 10] using PyTorch.
        x = torch.randn(10, 10)
        # Run the log_sinkhorn operator and exponentiate to get a doubly-stochastic matrix.
        y = torch.exp(decoders.log_sinkhorn(x, steps=10, temperature=1.0,
                                             zero_diagonal=False,
                                             noise_rng_key=None))
        # Verify that each row sums to 1.
        self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(10), atol=1e-4))
        # Verify that each column sums to 1.
        self.assertTrue(torch.allclose(y.sum(dim=-2), torch.ones(10), atol=1e-4))

    def test_log_sinkhorn_zero_diagonal(self):
        # Create a random tensor of shape [10, 10] using PyTorch.
        x = torch.randn(10, 10)
        # Run the log_sinkhorn operator with zero_diagonal=True.
        y = torch.exp(decoders.log_sinkhorn(x, steps=10, temperature=1.0,
                                             zero_diagonal=True,
                                             noise_rng_key=None))
        # Verify that each row sums to 1.
        self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(10), atol=1e-4))
        # Verify that each column sums to 1.
        self.assertTrue(torch.allclose(y.sum(dim=-2), torch.ones(10), atol=1e-4))
        # Verify that the diagonal elements are nearly zero.
        self.assertTrue(torch.allclose(torch.diag(y), torch.zeros(10), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
