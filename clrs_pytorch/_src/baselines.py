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

"""PyTorch implementation of CLRS baseline model."""

from typing import List, Union
import torch
from clrs_pytorch._src import nets, processors, samplers, specs

_Feedback = samplers.Feedback
_Spec = specs.Spec


class BaselineModel(torch.nn.Module):
  """Model implementation with selectable message passing algorithm."""

  def __init__(
      self,
      spec: Union[_Spec, List[_Spec]],
      dummy_trajectory: Union[List[_Feedback], _Feedback],
      device,
      processor_factory: processors.ProcessorFactory,
      hidden_dim: int = 32,
      encode_hints: bool = False,
      decode_hints: bool = True,
      encoder_init: str = 'default',
      use_lstm: bool = False,
      learning_rate: float = 0.005,
      grad_clip_max_norm: float = 0.0,
      checkpoint_path: str = '/tmp/clrs3',
      freeze_processor: bool = False,
      dropout_prob: float = 0.0,
      hint_teacher_forcing: float = 0.0,
      hint_repred_mode: str = 'soft',
      name: str = 'base_model',
      nb_msg_passing_steps: int = 1,
  ):
    """Constructor for BaselineModel.

    The model consists of encoders, processor and decoders. It can train
    and evaluate either a single algorithm or a set of algorithms; in the
    latter case, a single processor is shared among all the algorithms, while
    the encoders and decoders are separate for each algorithm.

    Args:
      spec: Either a single spec for one algorithm, or a list of specs for
        multiple algorithms to be trained and evaluated.
      dummy_trajectory: Either a single feedback batch, in the single-algorithm
        case, or a list of feedback batches, in the multi-algorithm case, that
        comply with the `spec` (or list of specs), to initialize network size.
      processor_factory: A callable that takes an `out_size` parameter
        and returns a processor (see `processors.py`).
      hidden_dim: Size of the hidden state of the model, i.e., size of the
        message-passing vectors.
      encode_hints: Whether to provide hints as model inputs.
      decode_hints: Whether to provide hints as model outputs.
      encoder_init: The initialiser type to use for the encoders.
      use_lstm: Whether to insert an LSTM after message passing.
      learning_rate: Learning rate for training.
      grad_clip_max_norm: if greater than 0, the maximum norm of the gradients.
      checkpoint_path: Path for loading/saving checkpoints.
      freeze_processor: If True, the processor weights will be frozen and
        only encoders and decoders (and, if used, the lstm) will be trained.
      dropout_prob: Dropout rate in the message-passing stage.
      hint_teacher_forcing: Probability of using ground-truth hints instead
        of predicted hints as inputs during training (only relevant if
        `encode_hints`=True)
      hint_repred_mode: How to process predicted hints when fed back as inputs.
        Only meaningful when `encode_hints` and `decode_hints` are True.
        Options are:
          - 'soft', where we use softmaxes for categoricals, pointers
              and mask_one, and sigmoids for masks. This will allow gradients
              to flow through hints during training.
          - 'hard', where we use argmax instead of softmax, and hard
              thresholding of masks. No gradients will go through the hints
              during training; even for scalar hints, which don't have any
              kind of post-processing, gradients will be stopped.
          - 'hard_on_eval', which is soft for training and hard for evaluation.

      nb_msg_passing_steps: Number of message passing steps per hint.
      debug: If True, the model run in debug mode, outputting all hidden state.

    Raises:
      ValueError: if `encode_hints=True` and `decode_hints=False`.
    """
    super().__init__()

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

    num_dims = []
    if isinstance(dummy_trajectory, _Feedback):
      assert len(spec) == 1
      dummy_trajectory = [dummy_trajectory]
    for traj in dummy_trajectory:
      nb_dims = {}
      for inp in traj.features.inputs:
        nb_dims[inp.name] = inp.data.shape[-1]
      for hint in traj.features.hints:
        nb_dims[hint.name] = hint.data.shape[-1]
      for outp in traj.outputs:
        nb_dims[outp.name] = outp.data.shape[-1]
      num_dims.append(nb_dims)
    
    self.net_fn = nets.Net(
          spec,
          hidden_dim,
          encode_hints,
          decode_hints,
          processor_factory,
          use_lstm,
          encoder_init,
          dropout_prob,
          hint_teacher_forcing,
          hint_repred_mode,
          num_dims,
          nb_msg_passing_steps,
          device,
      )


  def forward(
        self,
        feedback,
        algorithm_index: int,
        repred: bool = False,
        return_hints: bool = True,
        return_all_outputs: bool = False,
    ):
      """
      Perform a forward pass through the model.

      Args:
          feedback: The feedback data containing model features.
          algorithm_index: Index of the algorithm to be used.
          repred: If True, perform re-prediction.
          return_hints: Whether to return predicted hints.
          return_all_outputs: Whether to return all outputs from the network.

      Returns:
          A tuple (output_preds, hint_preds) containing the output predictions and hint predictions.
      """
      output_preds, hint_preds = self.net_fn(
          [feedback.features],
          repred=repred,
          algorithm_index=algorithm_index,
          return_hints=return_hints,
          return_all_outputs=return_all_outputs,
      )
      return output_preds, hint_preds
