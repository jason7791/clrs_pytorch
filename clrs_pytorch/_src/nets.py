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

"""Implementation of CLRS basic network."""
from typing import Dict, List, Optional, Tuple

from clrs_pytorch._src import decoders
from clrs_pytorch._src import encoders
from clrs_pytorch._src import probing
from clrs_pytorch._src import processors
from clrs_pytorch._src import samplers
from clrs_pytorch._src import specs

import jax
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Features = samplers.Features
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


@dataclass
class _MessagePassingScanState:
    hint_preds: torch.Tensor  
    output_preds: torch.Tensor  
    hiddens: torch.Tensor  
    lstm_state: Optional[torch.nn.Module] 

class Net(torch.nn.Module):
  """Building blocks (networks) used to encode and decode messages."""

  def __init__(
      self,
      spec: List[_Spec],
      hidden_dim: int,
      encode_hints: bool,
      decode_hints: bool,
      processor_factory: processors.ProcessorFactory,
      use_lstm: bool,
      encoder_init: str,
      dropout_prob: float,
      hint_teacher_forcing: float,
      hint_repred_mode='soft',
      nb_dims=None,
      nb_msg_passing_steps=1,
      device = torch.device('cuda:0'),
      name: str = 'net',
  ):
    """Constructs a `Net`."""
    super(Net, self).__init__()

    self._dropout_prob = dropout_prob
    self._hint_teacher_forcing = hint_teacher_forcing
    self._hint_repred_mode = hint_repred_mode
    self.spec = spec
    self.hidden_dim = hidden_dim
    self.encode_hints = encode_hints
    self.decode_hints = decode_hints
    self.processor_factory = processor_factory
    self.nb_dims = nb_dims
    self.use_lstm = use_lstm
    self.encoder_init = encoder_init
    self.nb_msg_passing_steps = nb_msg_passing_steps
    self.encoders, self.decoders = self._construct_encoders_decoders()
    self.processor = self.processor_factory(self.hidden_dim)
    self.device = device

    if self.use_lstm:
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.lstm_init = self.initialize_lstm_state
    else:
        self.lstm = None
        self.lstm_init = lambda x: 0

  def initialize_lstm_state(self, batch_size):
    h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
    c_0 = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
    return h_0, c_0
  
  def _msg_passing_step(self,
                        mp_state: _MessagePassingScanState,
                        i: int,
                        hints: List[_DataPoint],
                        repred: bool,
                        lengths: torch.Tensor,
                        batch_size: int,
                        nb_nodes: int,
                        inputs: _Trajectory,
                        first_step: bool,
                        spec: _Spec,
                        encs: Dict[str, List[nn.Module]],
                        decs: Dict[str, Tuple[nn.Module]],
                        return_hints: bool,
                        return_all_outputs: bool
                        ):
    if self.decode_hints and not first_step:
      assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
      hard_postprocess = (self._hint_repred_mode == 'hard' or
                          (self._hint_repred_mode == 'hard_on_eval' and repred))
      decoded_hint = decoders.postprocess(spec,
                                          mp_state.hint_preds,
                                          sinkhorn_temperature=0.1,
                                          sinkhorn_steps=25,
                                          hard=hard_postprocess)                                  
    if repred and self.decode_hints and not first_step:
      cur_hint = []
      for hint in decoded_hint:
        cur_hint.append(decoded_hint[hint])
    else:
      cur_hint = []
      needs_noise = (self.decode_hints and not first_step and
                     self._hint_teacher_forcing < 1.0)
      if needs_noise:
        # For noisy teacher forcing, choose which examples in the batch to force
        prob_tensor = torch.full((batch_size,), self._hint_teacher_forcing, device=self.device)

        # Use torch.bernoulli to sample binary values (0s and 1s) based on the probability
        force_mask = torch.bernoulli(prob_tensor)
      else:
        force_mask = None
      for hint in hints:
        hint_data = torch.tensor(hint.data, dtype=torch.long, device=self.device)[i]
        _, loc, typ = spec[hint.name]
        if needs_noise:
          if (typ == _Type.POINTER and
              decoded_hint[hint.name].type_ == _Type.SOFT_POINTER):
            # When using soft pointers, the decoded hints cannot be summarised
            # as indices (as would happen in hard postprocessing), so we need
            # to raise the ground-truth hint (potentially used for teacher
            # forcing) to its one-hot version.
            # hint_data = hint_data.long() 
            hint_data = torch.nn.functional.one_hot(hint_data, nb_nodes)
            typ = _Type.SOFT_POINTER
          hint_data = torch.where(_expand_to(force_mask.bool(), hint_data),
                                hint_data,
                                decoded_hint[hint.name].data)
        cur_hint.append(
            probing.DataPoint(
                name=hint.name, location=loc, type_=typ, data=hint_data))

    hiddens, output_preds_cand, hint_preds, lstm_state = self._one_step_pred(
        inputs, cur_hint, mp_state.hiddens,
        batch_size, nb_nodes, mp_state.lstm_state,
        spec, encs, decs, repred)

    if first_step:
      output_preds = output_preds_cand
    else:
      output_preds = {}
      for outp in mp_state.output_preds:
        is_not_done = _is_not_done_broadcast(lengths, i,
                                             output_preds_cand[outp], self.device)
        output_preds[outp] = is_not_done * output_preds_cand[outp] + (
            1.0 - is_not_done) * mp_state.output_preds[outp]

    new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
        hint_preds=hint_preds,
        output_preds=output_preds,
        hiddens=hiddens,
        lstm_state=lstm_state)
    # Save memory by not stacking unnecessary fields
    accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
        hint_preds=hint_preds if return_hints else None,
        output_preds=output_preds if return_all_outputs else None,
        hiddens=None, lstm_state=None)

    return new_mp_state, accum_mp_state

  def forward(self, features_list: List[_Features], repred: bool,
               algorithm_index: int,
               return_hints: bool,
               return_all_outputs: bool):
    """Process one batch of data.

    Args:
      features_list: A list of _Features objects, each with the inputs, hints
        and lengths for a batch o data corresponding to one algorithm.
        The list should have either length 1, at train/evaluation time,
        or length equal to the number of algorithms this Net is meant to
        process, at initialization.
      repred: False during training, when we have access to ground-truth hints.
        True in validation/test mode, when we have to use our own
        hint predictions.
      algorithm_index: Which algorithm is being processed. It can be -1 at
        initialisation (either because we are initialising the parameters of
        the module or because we are intialising the message-passing state),
        meaning that all algorithms should be processed, in which case
        `features_list` should have length equal to the number of specs of
        the Net. Otherwise, `algorithm_index` should be
        between 0 and `length(self.spec) - 1`, meaning only one of the
        algorithms will be processed, and `features_list` should have length 1.
      return_hints: Whether to accumulate and return the predicted hints,
        when they are decoded.
      return_all_outputs: Whether to return the full sequence of outputs, or
        just the last step's output.

    Returns:
      A 2-tuple with (output predictions, hint predictions)
      for the selected algorithm.
    """
    if algorithm_index == -1:
      algorithm_indices = range(len(features_list))
    else:
      algorithm_indices = [algorithm_index]
    assert len(algorithm_indices) == len(features_list)

    for algorithm_index, features in zip(algorithm_indices, features_list):
      inputs = features.inputs
      hints = features.hints
      lengths = features.lengths

      batch_size, nb_nodes = _data_dimensions(features)

      nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
      hiddens = torch.zeros((batch_size, nb_nodes, self.hidden_dim), device=self.device)

      if self.use_lstm:
        lstm_state = self.lstm_init(batch_size * nb_nodes)
        lstm_state = jax.tree_util.tree_map(
            lambda x, b=batch_size, n=nb_nodes: x.reshape(b, n, -1),
            lstm_state)
      else:
        lstm_state = None

      mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
          hint_preds=None, output_preds=None,
          hiddens=hiddens, lstm_state=lstm_state)

      common_args = dict(
          hints=hints,
          repred=repred,
          inputs=inputs,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
          lengths=lengths,
          spec=self.spec[algorithm_index],
          encs=self.encoders[algorithm_index],
          decs=self.decoders[algorithm_index],
          return_hints=return_hints,
          return_all_outputs=return_all_outputs,
          )
      mp_state, lean_mp_state = self._msg_passing_step(
          mp_state,
          i=0,
          first_step=True,
          **common_args)
    
      
      hint_preds_accum = {}
      output_preds_accum = {}
      if return_hints:
          for key, value in lean_mp_state.hint_preds.items():
              hint_preds_accum[key] = torch.zeros((nb_mp_steps, *value.shape), device=self.device)
              hint_preds_accum[key][0] = value

      if return_all_outputs:
          for key, value in lean_mp_state.output_preds.items():
              output_preds_accum[key] = torch.zeros((nb_mp_steps, *value.shape), device=self.device)
              output_preds_accum[key][0] = value

      for step in range(1, nb_mp_steps):
          mp_state, accum_mp_state = self._msg_passing_step(mp_state, step, first_step=False, **common_args)
          if return_hints:
              for key, value in accum_mp_state.hint_preds.items():
                  hint_preds_accum[key][step] = value

          if return_all_outputs:
              for key, value in accum_mp_state.output_preds.items():
                  output_preds_accum[key][step] = value

    def invert(d):
      """Dict of lists -> list of dicts."""
      if d: 
        return [dict(zip(d, i)) for i in zip(*d.values())]

    # Final output predictions
    output_preds = output_preds_accum if return_all_outputs else mp_state.output_preds

    hint_preds = invert(hint_preds_accum) if return_hints else None

    return output_preds, hint_preds

  def _construct_encoders_decoders(self):
    """Constructs encoders and decoders, separate for each algorithm."""
    encoders_ = nn.ModuleList()
    decoders_ = nn.ModuleList()
    enc_algo_idx = None

    for (algo_idx, spec) in enumerate(self.spec):
        enc = nn.ModuleDict()
        dec = nn.ModuleDict()

        for name, (stage, loc, t) in spec.items():
            if stage == _Stage.INPUT or (
                stage == _Stage.HINT and self.encode_hints):
                if name == specs.ALGO_IDX_INPUT_NAME:
                    if enc_algo_idx is None:
                        enc_algo_idx = nn.ModuleList([nn.LazyLinear(self.hidden_dim)])
                    enc[name] = enc_algo_idx
                else:
                    enc[name] = encoders.construct_encoders(
                        stage, loc, t, hidden_dim=self.hidden_dim,
                        init=self.encoder_init,
                        name=f'algo_{algo_idx}_{name}')

            if stage == _Stage.OUTPUT or (
                stage == _Stage.HINT and self.decode_hints):
                dec[name] = decoders.construct_decoders(
                    loc, t, hidden_dim=self.hidden_dim,
                    nb_dims=self.nb_dims[algo_idx][name],
                    name=f'algo_{algo_idx}_{name}')

        encoders_.append(enc)
        decoders_.append(dec)

    # Return as properly tracked nn.ModuleList
    return encoders_, decoders_


  def _one_step_pred(
      self,
      inputs: _Trajectory,
      hints: _Trajectory,
      hidden: _Array,
      batch_size: int,
      nb_nodes: int,
      lstm_state: Optional[torch.nn.Module],
      spec: _Spec,
      encs: Dict[str, List[nn.Module]],
      decs: Dict[str, Tuple[nn.Module]],
      repred: bool,
  ):
    """Generates one-step predictions."""
    # Initialize node features, edge features, graph features, and adjacency matrix in PyTorch
    node_fts = torch.zeros((batch_size, nb_nodes, self.hidden_dim), device=self.device)
    edge_fts = torch.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim), device=self.device)
    graph_fts = torch.zeros((batch_size, self.hidden_dim), device=self.device)

    # Create adjacency matrix with identity matrices repeated along the batch dimension
    adj_mat = torch.eye(nb_nodes, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode node/edge/graph features from inputs and (optionally) hints.
    trajectories = [inputs]
    if self.encode_hints:
      trajectories.append(hints)

    for trajectory in trajectories:
      for dp in trajectory:
        try:
          dp = encoders.preprocess(dp, nb_nodes)
          assert dp.type_ != _Type.SOFT_POINTER
          adj_mat = encoders.accum_adj_mat(dp, adj_mat)
          encoder = encs[dp.name]
          edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
          node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
          graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
        except Exception as e:
          raise Exception(f'Failed to process {dp}') from e

    # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nxt_hidden = hidden
    for _ in range(self.nb_msg_passing_steps):
      nxt_hidden, nxt_edge = self.processor(
          node_fts,
          edge_fts,
          graph_fts,
          adj_mat,
          nxt_hidden,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
      )

    if not repred:      # dropout only on training
      nxt_hidden = F.dropout(nxt_hidden, p=self._dropout_prob, training=self.training)

    if self.use_lstm:
      # lstm doesn't accept multiple batch dimensions (in our case, batch and
      # nodes), so we vmap over the (first) batch dimension.
      nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(nxt_hidden, lstm_state)
    else:
      nxt_lstm_state = None

    h_t = torch.cat([node_fts, hidden, nxt_hidden], dim=-1)
    if nxt_edge is not None:
      e_t = torch.cat([edge_fts, nxt_edge], dim=-1)
    else:
      e_t = edge_fts

    # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Decode features and (optionally) hints.
    hint_preds, output_preds = decoders.decode_fts(
        decoders=decs,
        spec=spec,
        h_t=h_t,
        adj_mat=adj_mat,
        edge_fts=e_t,
        graph_fts=graph_fts,
        inf_bias=self.processor.inf_bias,
        inf_bias_edge=self.processor.inf_bias_edge,
        repred=repred,
    )

    return nxt_hidden, output_preds, hint_preds, nxt_lstm_state

def _data_dimensions(features: _Features) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[:2]
  assert False


def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = torch.unsqueeze(x, -1)
  return x


def _is_not_done_broadcast(lengths, i, tensor, device):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device, dtype=torch.float32)
    is_not_done = (lengths > i + 1).float()
    while is_not_done.dim() < tensor.dim():
        is_not_done = is_not_done.unsqueeze(-1)
    return is_not_done
