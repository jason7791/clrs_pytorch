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

"""Unit tests for probing.py using only PyTorch.

This version creates expected tensors using torch.tensor and compares results
using PyTorch's testing utilities.
"""

from absl.testing import absltest
from clrs_pytorch._src import probing
import torch

# pylint: disable=invalid-name


class ProbingTest(absltest.TestCase):

  def test_array(self):
    A_pos = torch.tensor([1, 2, 0, 4, 3])
    expected = torch.tensor([2, 1, 1, 4, 0])
    out = probing.array(A_pos)
    self.assertTrue(torch.equal(expected, out))

  def test_array_cat(self):
    A = torch.tensor([2, 1, 0, 1, 1])
    expected = torch.tensor([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    out = probing.array_cat(A, 3)
    self.assertTrue(torch.equal(expected, out))

  def test_heap(self):
    A_pos = torch.tensor([1, 3, 5, 0, 7, 4, 2, 6])
    expected = torch.tensor([3, 1, 2, 1, 5, 1, 6, 3])
    out = probing.heap(A_pos, heap_size=6)
    self.assertTrue(torch.equal(expected, out))

  def test_graph(self):
    G = torch.tensor([
        [0.0, 7.0, -1.0, -3.9, 7.452],
        [0.0, 0.0, 133.0, 0.0, 9.3],
        [0.5, 0.1, 0.22, 0.55, 0.666],
        [7.0, 6.1, 0.2, 0.0, 0.0],
        [0.0, 3.0, 0.0, 1.0, 0.5]
    ])
    expected = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0]
    ], dtype=torch.float32)
    out = probing.graph(G)
    torch.testing.assert_close(expected, out)

  def test_mask_one(self):
    expected = torch.tensor([0, 0, 0, 1, 0])
    out = probing.mask_one(3, 5, device=torch.device('cpu'))
    self.assertTrue(torch.equal(expected, out))

  def test_strings_id(self):
    T_pos = torch.tensor([0, 1, 2, 3, 4])
    P_pos = torch.tensor([0, 1, 2])
    expected = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    out = probing.strings_id(T_pos, P_pos)
    self.assertTrue(torch.equal(expected, out))

  def test_strings_pair(self):
    pair_probe = torch.tensor([
        [0.5, 3.1, 9.1, 7.3],
        [1.0, 0.0, 8.0, 9.3],
        [0.1, 5.0, 0.0, 1.2]
    ], dtype=torch.float32)
    expected = torch.tensor([
        [0.0, 0.0, 0.0, 0.5, 3.1, 9.1, 7.3],
        [0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 9.3],
        [0.0, 0.0, 0.0, 0.1, 5.0, 0.0, 1.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    out = probing.strings_pair(pair_probe)
    torch.testing.assert_close(expected, out)

  def test_strings_pair_cat(self):
    pair_probe = torch.tensor([
        [0, 2, 1],
        [2, 2, 0]
    ])
    expected = torch.tensor([
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [1, 0, 0,  0],
            [0, 0, 1,  0],
            [0, 1, 0,  0],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 1,  0],
            [0, 0, 1,  0],
            [1, 0, 0,  0],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ],
    ])
    out = probing.strings_pair_cat(pair_probe, 3)
    self.assertTrue(torch.equal(expected, out))

  def test_strings_pi(self):
    T_pos = torch.tensor([0, 1, 2, 3, 4, 5])
    P_pos = torch.tensor([0, 1, 2, 3])
    pi = torch.tensor([3, 1, 0, 2])
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 9, 7, 6, 8])
    out = probing.strings_pi(T_pos, P_pos, pi)
    self.assertTrue(torch.equal(expected, out))

  def test_strings_pos(self):
    T_pos = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
    P_pos = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    expected = torch.tensor([
        0.0, 0.2, 0.4, 0.6, 0.8,
        0.0, 0.25, 0.5, 0.75
    ], dtype=torch.float32)
    out = probing.strings_pos(T_pos, P_pos)
    torch.testing.assert_close(expected, out)

  def test_strings_pred(self):
    T_pos = torch.tensor([0, 1, 2, 3, 4])
    P_pos = torch.tensor([0, 1, 2])
    expected = torch.tensor([0, 0, 1, 2, 3, 5, 5, 6])
    out = probing.strings_pred(T_pos, P_pos)
    self.assertTrue(torch.equal(expected, out))


class PermutationsTest(absltest.TestCase):

  def test_pointers_to_permutation(self):
    pointers = torch.tensor([2, 1, 1])
    perm, first = probing.predecessor_to_cyclic_predecessor_and_first(pointers)
    expected_perm = torch.tensor([[0, 0, 1],
                                  [1, 0, 0],
                                  [0, 1, 0]])
    expected_first = torch.tensor([0, 1, 0])
    torch.testing.assert_close(expected_perm, perm)
    torch.testing.assert_close(expected_first, first)

  def test_pointers_to_permutation_already_sorted(self):
    pointers = torch.tensor([0, 0, 1, 2, 3, 4])
    perm, first = probing.predecessor_to_cyclic_predecessor_and_first(pointers)
    expected_perm = torch.roll(torch.eye(6), 1, 0).to(torch.int64)
    expected_first = torch.eye(6)[0].to(torch.int64)
    torch.testing.assert_close(expected_perm, perm)
    torch.testing.assert_close(expected_first, first)


if __name__ == "__main__":
  absltest.main()
