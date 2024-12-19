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
"""decoders utilities."""

from typing import Dict, Optional

from clrs_pytorch._src import probing
from clrs_pytorch._src import specs
import haiku as hk
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type



def log_sinkhorn(x: _Array, steps: int, temperature: float, zero_diagonal: bool,
                 noise_rng_key: Optional[_Array]) -> _Array:
  """Sinkhorn operator in log space, to postprocess permutation pointer logits.

  Args:
    x: input of shape [..., n, n], a batch of square matrices.
    steps: number of iterations.
    temperature: temperature parameter (as temperature approaches zero, the
      output approaches a permutation matrix).
    zero_diagonal: whether to force the diagonal logits towards -inf.
    noise_rng_key: key to add Gumbel noise.

  Returns:
    Elementwise logarithm of a doubly-stochastic matrix (a matrix with
    non-negative elements whose rows and columns sum to 1).
  """
  assert x.ndim >= 2
  assert x.shape[-1] == x.shape[-2]
  if noise_rng_key is not None:
    # Add standard Gumbel noise (see https://arxiv.org/abs/1802.08665)
    # Generate uniform noise in PyTorch
    noise = torch.rand(x.shape, generator=noise_rng_key)  # noise_rng_key acts as a random seed

    # Apply the same log transformations as in the JAX code
    noise = -torch.log(-torch.log(noise + 1e-12) + 1e-12)
    x = x + noise
  x /= temperature
  if zero_diagonal:
    x = x - 1e6 * torch.eye(x.shape[-1])
  for _ in range(steps):
    x = jax.nn.log_softmax(x, axis=-1)
    x = jax.nn.log_softmax(x, axis=-2)
  return x

def construct_decoders(loc: str, t: str, hidden_dim: int, nb_dims: int, name: str):
    """Constructs decoders."""
    decoders = nn.ModuleList()
    if loc == _Location.NODE:
        # Node decoders.
        if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            decoders.append(nn.LazyLinear(1))
        elif t == _Type.CATEGORICAL:
            decoders.append(nn.LazyLinear(nb_dims))
        elif t in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
            decoders.extend([nn.LazyLinear(hidden_dim), nn.LazyLinear(hidden_dim), nn.LazyLinear(hidden_dim),
                  nn.LazyLinear(1)])
        else:
            raise ValueError(f"Invalid Type {t}")

    elif loc == _Location.EDGE:
        # Edge decoders.
        if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            decoders.extend([nn.LazyLinear(1), nn.LazyLinear(1), nn.LazyLinear(1)])
        elif t == _Type.CATEGORICAL:
            decoders.extend([nn.LazyLinear(nb_dims), nn.LazyLinear(nb_dims), nn.LazyLinear(nb_dims)])
        elif t == _Type.POINTER:
            decoders.extend([nn.LazyLinear(hidden_dim), nn.LazyLinear(hidden_dim),
                  nn.LazyLinear(hidden_dim), nn.LazyLinear(hidden_dim), nn.LazyLinear(1)])
        else:
            raise ValueError(f"Invalid Type {t}")

    elif loc == _Location.GRAPH:
        # Graph decoders.
        if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            decoders.extend([nn.LazyLinear(1), nn.LazyLinear(1)])
        elif t == _Type.CATEGORICAL:
            decoders.extend([nn.LazyLinear(nb_dims), nn.LazyLinear(nb_dims)])
        elif t == _Type.POINTER:
            decoders.extend([nn.LazyLinear(1), nn.LazyLinear(1),
                  nn.LazyLinear(1)])
        else:
            raise ValueError(f"Invalid Type {t}")

    else:
        raise ValueError(f"Invalid Location {loc}")

    return decoders

def construct_diff_decoders(name: str):
    """Constructs diff decoders."""
    decoders = nn.ModuleDict()  # Use ModuleDict to store layers

    # Define the linear layers and add them to the decoders
    decoders[_Location.NODE] = nn.LazyLinear( 1)  # Node decoder
    decoders[_Location.EDGE] = nn.ModuleList([
        nn.LazyLinear(1),  # Edge decoder 1
        nn.LazyLinear(1),  # Edge decoder 2
        nn.LazyLinear(1)   # Edge decoder 3
    ])
    decoders[_Location.GRAPH] = nn.ModuleList([
        nn.LazyLinear(1),  # Graph decoder 1
        nn.LazyLinear(1)   # Graph decoder 2
    ])

    return decoders


def postprocess(spec: _Spec, preds: Dict[str, _Array],
                sinkhorn_temperature: float,
                sinkhorn_steps: int,
                hard: bool) -> Dict[str, _DataPoint]:
    """Postprocesses decoder output.

    This is done on outputs in order to score performance, and on hints in
    order to score them but also in order to feed them back to the model.
    At scoring time, the postprocessing mode is "hard", logits will be
    arg-maxed and masks will be thresholded. However, for the case of the hints
    that are fed back in the model, the postprocessing can be hard or soft,
    depending on whether we want to let gradients flow through them or not.

    Args:
        spec: The spec of the algorithm whose outputs/hints we are postprocessing.
        preds: Output and/or hint predictions, as produced by decoders.
        sinkhorn_temperature: Parameter for the sinkhorn operator on permutation
          pointers.
        sinkhorn_steps: Parameter for the sinkhorn operator on permutation
          pointers.
        hard: whether to do hard postprocessing, which involves argmax for
          MASK_ONE, CATEGORICAL and POINTERS, thresholding for MASK, and stop
          gradient through for SCALAR. If False, soft postprocessing will be used,
          with softmax, sigmoid and gradients allowed.
    
    Returns:
        The postprocessed `preds`. In "soft" post-processing, POINTER types will
        change to SOFT_POINTER, so encoders know they do not need to be
        pre-processed before feeding them back in.
    """
    result = {}
    for name in preds.keys():
        _, loc, t = spec[name]
        new_t = t
        data = preds[name]
        
        if t == _Type.SCALAR:
            if hard:
                data = data.detach()  # Stop gradients
        elif t == _Type.MASK:
            if hard:
                data = (data > 0.0).float()
            else:
                data = torch.sigmoid(data)
        elif t in [_Type.MASK_ONE, _Type.CATEGORICAL]:
            cat_size = data.shape[-1]
            if hard:
                best = torch.argmax(data, dim=-1)
                data = F.one_hot(best, num_classes=cat_size).float()
            else:
                data = F.softmax(data, dim=-1)
        elif t == _Type.POINTER:
            if hard:
                data = torch.argmax(data, dim=-1).float()
            else:
                data = F.softmax(data, dim=-1)
                new_t = _Type.SOFT_POINTER
        elif t == _Type.PERMUTATION_POINTER:
            data = torch.exp(data)
            if hard:
                data = F.one_hot(torch.argmax(data, dim=-1), num_classes=data.shape[-1]).float()
        else:
            raise ValueError("Invalid type")
        
        result[name] = probing.DataPoint(
            name=name, location=loc, type_=new_t, data=data)

    return result

def decode_fts(
    decoders,
    spec: _Spec,
    h_t: _Array,
    adj_mat: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    inf_bias: bool,
    inf_bias_edge: bool,
    repred: bool,
):
    """Decodes node, edge, and graph features."""
    output_preds = {}
    hint_preds = {}

    for name in decoders:
        decoder = decoders[name]
        stage, loc, t = spec[name]

        if loc == _Location.NODE:
            preds = _decode_node_fts(decoder, t, h_t, edge_fts, adj_mat,
                                     inf_bias, repred)

        elif loc == _Location.EDGE:
            preds = _decode_edge_fts(decoder, t, h_t, edge_fts, adj_mat,
                                     inf_bias_edge)

        elif loc == _Location.GRAPH:
            preds = _decode_graph_fts(decoder, t, h_t, graph_fts)

        else:
            raise ValueError("Invalid output type")

        if stage == _Stage.OUTPUT:
            output_preds[name] = preds
        elif stage == _Stage.HINT:
            hint_preds[name] = preds
        else:
            raise ValueError(f"Found unexpected decoder {name}")

    return hint_preds, output_preds


def _decode_node_fts(decoders, t: str, h_t: _Array, edge_fts: _Array,
                     adj_mat: _Array, inf_bias: bool, repred: bool) -> _Array:
  """Decodes node features."""
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = torch.squeeze(decoders[0](h_t), -1)
  elif t == _Type.CATEGORICAL:
    preds = decoders[0](h_t)
  elif t in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
    p_1 = decoders[0](h_t)
    p_2 = decoders[1](h_t)
    p_3 = decoders[2](edge_fts)

    p_e = p_2.unsqueeze(-2) + p_3
    p_m = torch.maximum(p_1.unsqueeze(-2), p_e.permute(0, 2, 1, 3))

    preds = torch.squeeze(decoders[3](p_m), -1)

    if inf_bias:
      per_batch_min = torch.min(preds, dim=range(1, preds.ndim), keepdims=True)
      preds = torch.where(adj_mat > 0.5,
                        preds,
                        torch.minimum(-1.0, per_batch_min - 1.0))
    if t == _Type.PERMUTATION_POINTER:
      if repred:  # testing or validation, no Gumbel noise
        preds = log_sinkhorn(
            x=preds, steps=10, temperature=0.1,
            zero_diagonal=True, noise_rng_key=None)
      else:  # training, add Gumbel noise
        preds = log_sinkhorn(
            x=preds, steps=10, temperature=0.1,
            zero_diagonal=True, noise_rng_key=hk.next_rng_key())
  else:
    raise ValueError("Invalid output type")

  return preds


def _decode_edge_fts(decoders, t: str, h_t: _Array, edge_fts: _Array,
                     adj_mat: _Array, inf_bias_edge: bool) -> _Array:
  """Decodes edge features."""

  pred_1 = decoders[0](h_t)
  pred_2 = decoders[1](h_t)
  pred_e = decoders[2](edge_fts)
  pred = (torch.unsqueeze(pred_1, -2) + torch.unsqueeze(pred_2, -3) + pred_e)
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = torch.squeeze(pred, -1)
  elif t == _Type.CATEGORICAL:
    preds = pred
  elif t == _Type.POINTER:
    pred_2 = decoders[3](h_t)

    p_m = torch.maximum(torch.unsqueeze(pred, -2),
                      torch.unsqueeze(
                          torch.unsqueeze(pred_2, -3), -3))

    preds = torch.squeeze(decoders[4](p_m), -1)
  else:
    raise ValueError("Invalid output type")
  if inf_bias_edge and t in [_Type.MASK, _Type.MASK_ONE]:
    per_batch_min = torch.min(preds, dim=range(1, preds.ndim), keepdims=True)
    preds = torch.where(adj_mat > 0.5,
                      preds,
                      torch.minimum(-1.0, per_batch_min - 1.0))

  return preds


def _decode_graph_fts(decoders, t: str, h_t: _Array,
                      graph_fts: _Array) -> _Array:
  """Decodes graph features."""

  gr_emb, _ = torch.max(h_t, dim=-2)
  pred_n = decoders[0](gr_emb)
  pred_g = decoders[1](graph_fts)
  pred = pred_n + pred_g
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = torch.squeeze(pred, -1)
  elif t == _Type.CATEGORICAL:
    preds = pred
  elif t == _Type.POINTER:
    pred_2 = decoders[2](h_t)
    ptr_p = torch.unsqueeze(pred, 1) + pred_2.permute(0, 2, 1)
    preds = torch.squeeze(ptr_p, 1)
  else:
    raise ValueError("Invalid output type")

  return preds


def maybe_decode_diffs(
    diff_decoders,
    h_t: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    decode_diffs: bool,
) -> Optional[Dict[str, _Array]]:
  """Optionally decodes node, edge and graph diffs."""

  if decode_diffs:
    preds = {}
    node = _Location.NODE
    edge = _Location.EDGE
    graph = _Location.GRAPH
    preds[node] = _decode_node_diffs(diff_decoders[node], h_t)
    preds[edge] = _decode_edge_diffs(diff_decoders[edge], h_t, edge_fts)
    preds[graph] = _decode_graph_diffs(diff_decoders[graph], h_t, graph_fts)

  else:
    preds = None

  return preds


def _decode_node_diffs(decoders, h_t: _Array) -> _Array:
  """Decodes node diffs."""
  return torch.squeeze(decoders(h_t), -1)


def _decode_edge_diffs(decoders, h_t: _Array, edge_fts: _Array) -> _Array:
  """Decodes edge diffs."""

  e_pred_1 = decoders[0](h_t)
  e_pred_2 = decoders[1](h_t)
  e_pred_e = decoders[2](edge_fts)
  preds = torch.squeeze(
      torch.unsqueeze(e_pred_1, -1) + torch.unsqueeze(e_pred_2, -2) + e_pred_e,
      -1,
  )

  return preds


def _decode_graph_diffs(decoders, h_t: _Array, graph_fts: _Array) -> _Array:
  """Decodes graph diffs."""

  gr_emb = torch.max(h_t, dim=-2)
  g_pred_n = decoders[0](gr_emb)
  g_pred_g = decoders[1](graph_fts)
  preds = torch.squeeze(g_pred_n + g_pred_g, -1)

  return preds
