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

"""Probing utilities.

The dataflow for an algorithm is represented by `(stage, loc, type, data)`
"probes" that are valid under that algorithm's spec (see `specs.py`).

When constructing probes, it is convenient to represent these fields in a nested
format (`ProbesDict`) to facilitate efficient context-based look-up.
"""

import functools
from typing import Dict, List, Tuple, Union

import attr
from clrs_pytorch._src import specs
import torch
import tensorflow as tf
import jax

_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type
_OutputClass = specs.OutputClass
_Array = torch.Tensor
_Data = Union[_Array, List[_Array]]
_DataOrType = Union[_Data, str]

ProbesDict = Dict[
    str, Dict[str, Dict[str, Dict[str, _DataOrType]]]]


def _convert_to_str(element):
  if isinstance(element, tf.Tensor):
    return element.numpy().decode('utf-8')
  elif isinstance(element, (torch.Tensor, bytes)):
    # If the tensor contains a single element, return its Python scalar.
    if isinstance(element, torch.Tensor) and element.numel() == 1:
      return str(element.item())
    elif isinstance(element, bytes):
      return element.decode('utf-8')
    else:
      return str(element)
  else:
    return element


@jax.tree_util.register_pytree_node_class
@attr.define
class DataPoint:
  """Describes a data point."""
  _name: str
  _location: str
  _type_: str
  data: _Array

  @property
  def name(self):
    return _convert_to_str(self._name)

  @property
  def location(self):
    return _convert_to_str(self._location)

  @property
  def type_(self):
    return _convert_to_str(self._type_)

  def __repr__(self):
    s = f'DataPoint(name="{self.name}",\tlocation={self.location},\t'
    return s + f'type={self.type_},\tdata=Array{self.data.shape})'

  def tree_flatten(self):
    data = (self.data,)
    meta = (self.name, self.location, self.type_)
    return data, meta

  @classmethod
  def tree_unflatten(cls, meta, data):
    name, location, type_ = meta
    subdata, = data
    return DataPoint(name, location, type_, subdata)


class ProbeError(Exception):
  pass


def initialize(spec: specs.Spec) -> ProbesDict:
  """Initializes an empty `ProbesDict` corresponding with the provided spec."""
  probes = {}
  for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    probes[stage] = {}
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
      probes[stage][loc] = {}
  for name in spec:
    stage, loc, t = spec[name]
    probes[stage][loc][name] = {}
    probes[stage][loc][name]['data'] = []
    probes[stage][loc][name]['type_'] = t
  return probes


def push(probes: ProbesDict, stage: str, next_probe):
  """Pushes a probe into an existing `ProbesDict`."""
  for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
    for name in probes[stage][loc]:
      if name not in next_probe:
        raise ProbeError(f'Missing probe for {name}.')
      if isinstance(probes[stage][loc][name]['data'], torch.Tensor):
        raise ProbeError('Attempting to push to finalized `ProbesDict`.')
      probes[stage][loc][name]['data'].append(next_probe[name])


def finalize(probes: ProbesDict):
  """Finalizes a `ProbesDict` by stacking/squeezing the `data` field."""
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
      for name in probes[stage][loc]:
        # Check if already finalized.
        if isinstance(probes[stage][loc][name]['data'], torch.Tensor):
          raise ProbeError('Attempting to re-finalize a finalized `ProbesDict`.')
        data_list = []
        for x in probes[stage][loc][name]['data']:
          if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=device)
          data_list.append(x)
        if stage == _Stage.HINT:
          probes[stage][loc][name]['data'] = torch.stack(data_list)
        else:
          probes[stage][loc][name]['data'] = torch.squeeze(torch.stack(data_list))


def split_stages(
    probes: ProbesDict,
    spec: specs.Spec,
) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint]]:
  """Splits contents of `ProbesDict` into `DataPoint`s by stage."""
  inputs = []
  outputs = []
  hints = []
  for name in spec:
    stage, loc, t = spec[name]
    if stage not in probes:
      raise ProbeError(f'Missing stage {stage}.')
    if loc not in probes[stage]:
      raise ProbeError(f'Missing location {loc}.')
    if name not in probes[stage][loc]:
      raise ProbeError(f'Missing probe {name}.')
    if 'type_' not in probes[stage][loc][name]:
      raise ProbeError(f'Probe {name} missing attribute `type_`.')
    if 'data' not in probes[stage][loc][name]:
      raise ProbeError(f'Probe {name} missing attribute `data`.')
    if t != probes[stage][loc][name]['type_']:
      raise ProbeError(f'Probe {name} of incorrect type {t}.')
    data = probes[stage][loc][name]['data']
    if not isinstance(data, torch.Tensor):
      raise ProbeError(f'Invalid `data` for probe "{name}". Did you forget to call `finalize`?')
    if t in [_Type.MASK, _Type.MASK_ONE, _Type.CATEGORICAL]:
      if not (((data == 0) | (data == 1) | (data == -1)).all()):
        raise ProbeError(f'0|1|-1 `data` for probe "{name}"')
      if t in [_Type.MASK_ONE, _Type.CATEGORICAL] and not torch.all(torch.sum(torch.abs(data), -1) == 1):
        raise ProbeError(f'Expected one-hot `data` for probe "{name}"')
    dim_to_expand = 1 if stage == _Stage.HINT else 0
    data_point = DataPoint(name=name, location=loc, type_=t,
                           data=data.unsqueeze(dim_to_expand))
    if stage == _Stage.INPUT:
      inputs.append(data_point)
    elif stage == _Stage.OUTPUT:
      outputs.append(data_point)
    else:
      hints.append(data_point)
  return inputs, outputs, hints


def array(A_pos: torch.tensor) -> torch.tensor:
  """Constructs an `array` probe."""
  if not isinstance(A_pos, torch.Tensor):
    A_pos = torch.as_tensor(A_pos)
  probe = torch.arange(A_pos.shape[0], device=A_pos.device)
  for i in range(1, A_pos.shape[0]):
    probe[A_pos[i].item()] = A_pos[i - 1]
  return probe


def array_cat(A: torch.tensor, n: int) -> torch.tensor:
  """Constructs an `array_cat` probe."""
  if not isinstance(A, torch.Tensor):
    A = torch.as_tensor(A)
  assert n > 0
  probe = torch.zeros((A.shape[0], n), device=A.device)
  for i in range(A.shape[0]):
    probe[i, A[i].item()] = 1
  return probe


def heap(A_pos: torch.tensor, heap_size: int) -> torch.tensor:
  """Constructs a `heap` probe."""
  if not isinstance(A_pos, torch.Tensor):
    A_pos = torch.as_tensor(A_pos)
  assert heap_size > 0
  probe = torch.arange(A_pos.shape[0], device=A_pos.device)
  for i in range(1, heap_size):
    probe[A_pos[i].item()] = A_pos[(i - 1) // 2]
  return probe


def graph(A: torch.tensor) -> torch.tensor:
  """Constructs a `graph` probe."""
  if not isinstance(A, torch.Tensor):
    A = torch.as_tensor(A)
  probe = (A != 0).float()
  probe = ((A + torch.eye(A.shape[0], device=A.device)) != 0).float()
  return probe


def mask_one(i: int, n: int, 
             device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ) -> torch.Tensor:
    """Constructs a `mask_one` probe on the given device."""
    assert n > i
    probe = torch.zeros(n, device=device)
    probe[i] = 1
    return probe


def strings_id(T_pos: torch.tensor, P_pos: torch.tensor) -> torch.tensor:
  """Constructs a `strings_id` probe."""
  if not isinstance(T_pos, torch.Tensor):
    T_pos = torch.as_tensor(T_pos)
  if not isinstance(P_pos, torch.Tensor):
    P_pos = torch.as_tensor(P_pos)
  device = T_pos.device
  probe_T = torch.zeros(T_pos.shape[0], device=device)
  probe_P = torch.ones(P_pos.shape[0], device=device)
  return torch.cat([probe_T, probe_P])


def strings_pair(pair_probe: torch.tensor) -> torch.tensor:
  """Constructs a `strings_pair` probe."""
  if not isinstance(pair_probe, torch.Tensor):
    pair_probe = torch.as_tensor(pair_probe)
  n = pair_probe.shape[0]
  m = pair_probe.shape[1]
  probe_ret = torch.zeros((n + m, n + m), dtype=pair_probe.dtype, device=pair_probe.device)
  for i in range(n):
    for j in range(m):
      probe_ret[i, j + n] = pair_probe[i, j]
  return probe_ret


def strings_pair_cat(pair_probe: torch.tensor, nb_classes: int) -> torch.tensor:
  """Constructs a `strings_pair_cat` probe."""
  if not isinstance(pair_probe, torch.Tensor):
    pair_probe = torch.as_tensor(pair_probe)
  assert nb_classes > 0
  n = pair_probe.shape[0]
  m = pair_probe.shape[1]
  probe_ret = torch.zeros((n + m, n + m, nb_classes + 1), device=pair_probe.device)
  for i in range(n):
    for j in range(m):
      probe_ret[i, j + n, int(pair_probe[i, j].item())] = _OutputClass.POSITIVE
  for i_1 in range(n):
    for i_2 in range(n):
      probe_ret[i_1, i_2, nb_classes] = _OutputClass.MASKED
  for j_1 in range(m):
    for x in range(n + m):
      probe_ret[j_1 + n, x, nb_classes] = _OutputClass.MASKED
  return probe_ret


def strings_pi(T_pos: torch.tensor, P_pos: torch.tensor, pi: torch.tensor) -> torch.tensor:
  """Constructs a `strings_pi` probe."""
  if not isinstance(T_pos, torch.Tensor):
    T_pos = torch.as_tensor(T_pos)
  if not isinstance(P_pos, torch.Tensor):
    P_pos = torch.as_tensor(P_pos)
  if not isinstance(pi, torch.Tensor):
    pi = torch.as_tensor(pi)
  total = T_pos.shape[0] + P_pos.shape[0]
  probe = torch.arange(total, device=T_pos.device)
  for j in range(P_pos.shape[0]):
    idx = T_pos.shape[0] + P_pos[j].item()
    probe[idx] = T_pos.shape[0] + pi[P_pos[j].item()]
  return probe


def strings_pos(T_pos: torch.tensor, P_pos: torch.tensor) -> torch.tensor:
  """Constructs a `strings_pos` probe."""
  if not isinstance(T_pos, torch.Tensor):
    T_pos = torch.as_tensor(T_pos)
  if not isinstance(P_pos, torch.Tensor):
    P_pos = torch.as_tensor(P_pos)
  probe_T = T_pos.clone().float() / T_pos.shape[0]
  probe_P = P_pos.clone().float() / P_pos.shape[0]
  return torch.cat([probe_T, probe_P])


def strings_pred(T_pos: torch.tensor, P_pos: torch.tensor) -> torch.tensor:
  """Constructs a `strings_pred` probe."""
  if not isinstance(T_pos, torch.Tensor):
    T_pos = torch.as_tensor(T_pos)
  if not isinstance(P_pos, torch.Tensor):
    P_pos = torch.as_tensor(P_pos)
  total = T_pos.shape[0] + P_pos.shape[0]
  probe = torch.arange(total, device=T_pos.device)
  for i in range(1, T_pos.shape[0]):
    probe[T_pos[i].item()] = T_pos[i - 1]
  for j in range(1, P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j].item()] = T_pos.shape[0] + P_pos[j - 1].item()
  return probe


def predecessor_pointers_to_permutation_matrix(pointers: torch.tensor) -> torch.tensor:
  """Converts predecessor pointers to a permutation matrix."""
  if not isinstance(pointers, torch.Tensor):
    pointers = torch.as_tensor(pointers)
  nb_nodes = pointers.shape[-1]
  pointers_one_hot = torch.nn.functional.one_hot(pointers, nb_nodes)
  last = pointers_one_hot.sum(-2).argmin()
  perm = torch.zeros((nb_nodes, nb_nodes), device=pointers.device, dtype=pointers.dtype)
  for i in range(nb_nodes - 1, -1, -1):
    perm += (
        torch.nn.functional.one_hot(torch.tensor(i, device=pointers.device), nb_nodes)[..., None] *
        torch.nn.functional.one_hot(last, nb_nodes)
    )
    last = pointers[last].item()
  return perm


def permutation_matrix_to_predecessor_pointers(perm: torch.tensor) -> torch.tensor:
  """Converts a permutation matrix to predecessor pointers."""
  if not isinstance(perm, torch.Tensor):
    perm = torch.as_tensor(perm)
  nb_nodes = perm.shape[-1]
  pointers = torch.zeros(nb_nodes, dtype=torch.int64, device=perm.device)
  idx = perm.argmax(-1)
  pointers += idx[0].item() * torch.nn.functional.one_hot(idx[0], nb_nodes)
  for i in range(1, nb_nodes):
    pointers += idx[i - 1].item() * torch.nn.functional.one_hot(idx[i], nb_nodes)
  pointers = torch.minimum(pointers, torch.tensor(nb_nodes - 1, device=perm.device))
  return pointers


def predecessor_to_cyclic_predecessor_and_first(
    pointers: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
  """Converts predecessor pointers to cyclic predecessor + first node mask."""
  if not isinstance(pointers, torch.Tensor):
    pointers = torch.as_tensor(pointers)
  pointers = pointers.to(dtype=torch.long)
  nb_nodes = pointers.shape[-1]
  pointers_one_hot = torch.nn.functional.one_hot(pointers, nb_nodes)
  last = pointers_one_hot.sum(-2).argmin()
  first = pointers_one_hot.diagonal().argmax()
  mask = torch.nn.functional.one_hot(first, nb_nodes)
  pointers_one_hot += mask[..., None] * torch.nn.functional.one_hot(last, nb_nodes)
  pointers_one_hot -= mask[..., None] * mask
  return pointers_one_hot, mask
