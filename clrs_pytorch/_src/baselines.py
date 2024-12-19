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
    super().__init__()

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

    self.nb_dims = []
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
      self.nb_dims.append(nb_dims)

    self.net_fn = nets.Net(
      spec, hidden_dim, encode_hints, decode_hints,
      processor_factory, use_lstm, encoder_init,
      dropout_prob, hint_teacher_forcing,
      hint_repred_mode,
      self.nb_dims, nb_msg_passing_steps)


  def forward(self, feedback, algorithm_index, repred=False, return_hints=True, return_all_outputs=False):
    output_preds, hint_preds = self.net_fn( [feedback.features],
        repred=repred,
        algorithm_index=algorithm_index,
        return_hints=return_hints,
        return_all_outputs=return_all_outputs)
    return output_preds, hint_preds
