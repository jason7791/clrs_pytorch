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

"""JAX implementation of CLRS basic network."""

import functools

from typing import Dict, List, Optional, Tuple


from clrs_pytorch._src import decoders
from clrs_pytorch._src import encoders
from clrs_pytorch._src import probing
from clrs_pytorch._src import processors
from clrs_pytorch._src import samplers
from clrs_pytorch._src import specs

import haiku as hk
import jax
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
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
    lstm_state: Optional[torch.nn.Module]  # For LSTM state

    def accumulate(self, other):
        """Accumulates fields from another MessagePassingScanState object."""
        
        # Accumulate `hint_preds` (assuming both are dicts)
        if other.hint_preds:
            if self.hint_preds is None:
                # Initialize the dictionary if it's None
                self.hint_preds = {key: value.unsqueeze(0) for key, value in other.hint_preds.items()}
            else:
                # Loop through each key in the dictionary and concatenate the values
                for key, value in other.hint_preds.items():
                    if key in self.hint_preds:
                        self.hint_preds[key] = torch.cat([self.hint_preds[key], value.unsqueeze(0)], dim=0)
                    else:
                        self.hint_preds[key] = value.unsqueeze(0)

        # Accumulate `output_preds` (assuming both are dicts)
        if other.output_preds:
            if self.output_preds is None:
                # Initialize the dictionary if it's None
                self.output_preds = {key: value.unsqueeze(0) for key, value in other.output_preds.items()}
            else:
                # Loop through each key in the dictionary and concatenate the values
                for key, value in other.output_preds.items():
                    if key in self.output_preds:
                        self.output_preds[key] = torch.cat([self.output_preds[key], value.unsqueeze(0)], dim=0)
                    else:
                        self.output_preds[key] = value.unsqueeze(0)

@dataclass
class _MessagePassingStateChunked:
    inputs: torch.Tensor 
    hints: torch.Tensor 
    is_first: torch.Tensor 
    hint_preds: torch.Tensor 
    hiddens: torch.Tensor  
    lstm_state: Optional[torch.nn.Module]  # For LSTM state


class Net(nn.Module):
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
      name: str = 'net',
  ):
    """Constructs a `Net`."""
    # super().__init__(name=name)
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
    self.layers = nn.ModuleList()  # Register a list of layers

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
        prob_tensor = torch.full((batch_size,), self._hint_teacher_forcing)

        # Use torch.bernoulli to sample binary values (0s and 1s) based on the probability
        force_mask = torch.bernoulli(prob_tensor)
      else:
        force_mask = None
      for hint in hints:
        hint_data = torch.tensor(hint.data)[i]
        _, loc, typ = spec[hint.name]
        if needs_noise:
          if (typ == _Type.POINTER and
              decoded_hint[hint.name].type_ == _Type.SOFT_POINTER):
            # When using soft pointers, the decoded hints cannot be summarised
            # as indices (as would happen in hard postprocessing), so we need
            # to raise the ground-truth hint (potentially used for teacher
            # forcing) to its one-hot version.
            hint_data = hint_data.long() 
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
                                             output_preds_cand[outp])
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

    # Complying to jax.scan, the first returned value is the state we carry over
    # the second value is the output that will be stacked over steps.
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

    self.encoders, self.decoders = self._construct_encoders_decoders()
    self.processor = self.processor_factory(self.hidden_dim)

    # Optional LSTM construction in PyTorch
    if self.use_lstm:
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,  # Specify input size for LSTM
            hidden_size=self.hidden_dim,  # Hidden size for LSTM
            batch_first=True,             # Optional: Set batch first if batch is the leading dimension
            num_layers=1,                 # Number of LSTM layers (default is 1)
            bidirectional=False           # Set False to match Haiku's unidirectional LSTM
        )
        
        # Define a function to initialize the LSTM state (hidden and cell states)
        def lstm_init(batch_size, device):
            # Create initial hidden state and cell state as zeros
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)  # Hidden state
            c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)  # Cell state
            return (h0, c0)
    else:
        self.lstm = None
        # Return a default value for the LSTM state
        lstm_init = lambda batch_size, device: (0, 0)

    for algorithm_index, features in zip(algorithm_indices, features_list):
      inputs = features.inputs
      hints = features.hints
      lengths = features.lengths

      batch_size, nb_nodes = _data_dimensions(features)

      nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
      hiddens = torch.zeros((batch_size, nb_nodes, self.hidden_dim))

      if self.use_lstm:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_state = lstm_init(batch_size * nb_nodes, device)
        lstm_state = jax.tree_util.tree_map(
            lambda x, b=batch_size, n=nb_nodes: x.reshape(b, n, -1),
            lstm_state)
      else:
        lstm_state = None

      mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
          hint_preds=None, output_preds=None,
          hiddens=hiddens, lstm_state=lstm_state)

      # Do the first step outside of the scan because it has a different
      # computation graph.
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
    
      scan_fn = functools.partial(
          self._msg_passing_step,
          first_step=False,
          **common_args)


      # Initialize the output states with the initial state
      output_mp_state = mp_state  # Initialize with the starting mp_state
      accum_mp_state = _MessagePassingScanState(
          hint_preds={key: value.unsqueeze(0) for key, value in lean_mp_state.hint_preds.items()} if lean_mp_state.hint_preds is not None else None,
          output_preds={key: value.unsqueeze(0) for key, value in lean_mp_state.output_preds.items()} if lean_mp_state.output_preds is not None else None,
          hiddens=lean_mp_state.hiddens,  # Copy hiddens without modification
          lstm_state=lean_mp_state.lstm_state  # Copy LSTM state without modification
      )

      # Loop through the range of steps
      for step in range(1, nb_mp_steps):
          output_mp_state, current_accum = scan_fn(output_mp_state, step)
          accum_mp_state.accumulate(current_accum)


    # We only return the last algorithm's output. That's because
    # the output only matters when a single algorithm is processed; the case
    # `algorithm_index==-1` (meaning all algorithms should be processed)
    # is used only to init parameters.
    def accumulate_fn(init, tail):
        return torch.cat([init.unsqueeze(0), tail], dim=0)


    def tree_map_message_passing_state(fn, mp_state1, mp_state2):
        """Applies a function to the corresponding fields in two MessagePassingScanState objects."""
        return _MessagePassingScanState(
            hint_preds=fn(mp_state1.hint_preds, mp_state2.hint_preds),
            output_preds=fn(mp_state1.output_preds, mp_state2.output_preds),
            hiddens=fn(mp_state1.hiddens, mp_state2.hiddens),
            lstm_state=fn(mp_state1.lstm_state, mp_state2.lstm_state)
        )

    # Now apply it to lean_mp_state and current_accum
    # accum_mp_state = tree_map_message_passing_state(accumulate_fn, lean_mp_state, accum_mp_state)

    print("ACCUM hint preds", nb_mp_steps)
    if(accum_mp_state.hint_preds):
      print(type(accum_mp_state.hint_preds), len(accum_mp_state.hint_preds), accum_mp_state.hint_preds.keys(), accum_mp_state.hint_preds['d'].shape)
    def invert(d):
      """Dict of lists -> list of dicts."""
      if d: 
        return [dict(zip(d, i)) for i in zip(*d.values())]

    if return_all_outputs:
      output_preds = {k: torch.stack(v)
                      for k, v in accum_mp_state.output_preds.items()}
    else:
      output_preds = output_mp_state.output_preds
    hint_preds = invert(accum_mp_state.hint_preds)

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
          # Build input encoders.
          if name == specs.ALGO_IDX_INPUT_NAME:
            if enc_algo_idx is None:
                enc_algo_idx = [nn.Linear(self.hidden_dim, self.hidden_dim)]
            enc[name] = enc_algo_idx
          else:
            enc[name] = encoders.construct_encoders(
                stage, loc, t, hidden_dim=self.hidden_dim,
                init=self.encoder_init,
                name=f'algo_{algo_idx}_{name}')

        if stage == _Stage.OUTPUT or (
            stage == _Stage.HINT and self.decode_hints):
          # Build output decoders.
          dec[name] = decoders.construct_decoders(
              loc, t, hidden_dim=self.hidden_dim,
              nb_dims=self.nb_dims[algo_idx][name],
              name=f'algo_{algo_idx}_{name}')
      encoders_.append(enc)
      decoders_.append(dec)

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

    # Initialise empty node/edge/graph features and adjacency matrix.
    # Initialize node features, edge features, graph features, and adjacency matrix in PyTorch
    node_fts = torch.zeros((batch_size, nb_nodes, self.hidden_dim))
    edge_fts = torch.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = torch.zeros((batch_size, self.hidden_dim))

    # Create adjacency matrix with identity matrices repeated along the batch dimension
    adj_mat = torch.eye(nb_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

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


def _data_dimensions_chunked(features: _FeaturesChunked) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[1:3]
  assert False


def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = torch.unsqueeze(x, -1)
  return x


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):  # pytype: disable=attribute-error  # numpy-scalars
    is_not_done = torch.unsqueeze(torch.Tensor(is_not_done), -1)
  return is_not_done
