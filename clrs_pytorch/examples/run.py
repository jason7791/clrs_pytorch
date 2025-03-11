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
"""Run training of one or more algorithmic tasks from clrs_pytorch."""

import functools
import os
import shutil
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import jax
import tensorflow as tf
import requests
from absl import app, flags, logging

import clrs_pytorch
from clrs_pytorch._src import specs, losses, samplers, decoders

# Type aliases for clarity.
_Feedback = samplers.Feedback
_Location = specs.Location

# -------------------------- Flag Definitions --------------------------
flags.DEFINE_list('algorithms', ['graham_scan'], 'Algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Training sizes to use. A size of -1 means use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Needle length for training and validation (not testing) in string matching. '
                     'A negative value randomizes the length between 1 and the absolute value. '
                     'A value of 0 means use always 1/4 of the haystack length (default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed.')

flags.DEFINE_boolean('random_pos', True, 'Randomize the pos input common to all algorithms.')
flags.DEFINE_boolean('enforce_permutations', True, 'Enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True, 'Convert fixed pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_boolean('chunked_training', False, 'Use chunking for training.')
flags.DEFINE_integer('chunk_length', 16, 'Time chunk length for training (if chunked_training is True).')
flags.DEFINE_integer('train_steps', 500, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Testing frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors.')
flags.DEFINE_integer('nb_msg_passing_steps', 1, 'Number of message passing steps per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('grad_clip_max_norm', 1.0, 'Gradient clipping norm (0.0 disables clipping).')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded during training '
                   'instead of predicted hints (only pertinent in encoded_decoded modes).')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'Hint mode: encoded_decoded (hardest, default), decoded_only, or none.')
flags.DEFINE_enum('hint_repred_mode', 'soft',
                  ['soft', 'hard', 'hard_on_eval'],
                  'Mode for processing predicted hints: soft, hard, or hard_on_eval.')
flags.DEFINE_boolean('use_ln', True, 'Use layer normalization in the processor.')
flags.DEFINE_boolean('use_lstm', False, 'Insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8, 'Number of triplet features to compute.')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initializer to use for the encoders.')
flags.DEFINE_enum('processor_type', 'mpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', '/Users/jasonwjh/Documents/clrs_pytorch/clrs_checkpoints/checkpoint_shortest_paths.pth',
                    'Path to save checkpoints.')
flags.DEFINE_string('performance_path', '/Users/jasonwjh/Documents/clrs_pytorch/clrs_performance/performance_shortest_paths.json',
                    'Path to save performance results.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30', 'Path where the dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False, 'Freeze the processor of the model.')
flags.DEFINE_boolean('resume', False, 'Resume training from the last saved checkpoint if available.')

FLAGS = flags.FLAGS

# Algorithms that require converting pred_h hints to inputs.
PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march'
]

# ------------------------- Helper Functions -------------------------
def unpack(v):
    """Unpack a scalar value from a tensor or return the original value."""
    try:
        return v.item()
    except (AttributeError, ValueError):
        return v

def _iterate_sampler(sampler, batch_size):
    """Yield batches indefinitely from a sampler."""
    while True:
        yield sampler.next(batch_size)

def _maybe_download_dataset(dataset_path: str) -> str:
    """Download the CLRS30 dataset if it is not already present."""
    dataset_folder = os.path.join(dataset_path, clrs_pytorch.get_clrs_folder())
    if os.path.isdir(dataset_folder):
        logging.info('Dataset found at %s. Skipping download.', dataset_folder)
        return dataset_folder

    logging.info('Dataset not found in %s. Downloading...', dataset_folder)
    clrs_url = clrs_pytorch.get_dataset_gcp_url()
    response = requests.get(clrs_url, allow_redirects=True)
    clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
    os.makedirs(dataset_folder)
    with open(clrs_file, 'wb') as f:
        f.write(response.content)
    shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
    os.remove(clrs_file)
    return dataset_folder

def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
    """Create a sampler with the specified options.

    Args:
      length: Sample size (number of nodes). A length of -1 uses the benchmark dataset.
      rng: Numpy random state.
      algorithm: Name of the algorithm to sample from.
      split: One of 'train', 'val', or 'test'.
      batch_size: Batch size.
      multiplier: Multiplier for the number of samples (negative for infinite samples).
      randomize_pos: Randomize the `pos` input.
      enforce_pred_as_input: Convert fixed pred_h hints to inputs.
      enforce_permutations: Enforce permutation pointers.
      chunked: Whether to use chunking.
      chunk_length: Time chunk length if chunked is True.
      sampler_kwargs: Extra arguments for the sampler.
    Returns:
      A tuple (sampler, num_samples, spec).
    """
    if length < 0:
        dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
        sampler, num_samples, spec = clrs_pytorch.create_dataset(
            folder=dataset_folder,
            algorithm=algorithm,
            batch_size=batch_size,
            split=split
        )
        sampler = sampler.as_numpy_iterator()
    else:
        num_samples = clrs_pytorch.CLRS30[split]['num_samples'] * multiplier
        sampler, spec = clrs_pytorch.build_sampler(
            algorithm,
            seed=rng.randint(2**32),
            num_samples=num_samples,
            length=length,
            **sampler_kwargs
        )
        sampler = _iterate_sampler(sampler, batch_size)

    if randomize_pos:
        sampler = clrs_pytorch.process_random_pos(sampler, rng)
    if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
        spec, sampler = clrs_pytorch.process_pred_as_input(spec, sampler)
    spec, sampler = clrs_pytorch.process_permutations(spec, sampler, enforce_permutations)
    if chunked:
        sampler = clrs_pytorch.chunkify(sampler, chunk_length)
    return sampler, num_samples, spec

def make_multi_sampler(sizes: List[int], rng: Any, **kwargs):
    """Create a sampler with cycling sample sizes."""
    samplers_list = []
    tot_samples = 0
    for length in sizes:
        sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
        samplers_list.append(sampler)
        tot_samples += num_samples

    def cycle_samplers():
        while True:
            for s in samplers_list:
                yield next(s)
    return cycle_samplers(), tot_samples, spec

def _concat(dps, axis: int):
    return jax.tree_util.tree_map(lambda *x: torch.cat(x, axis), *dps)

def collect_and_eval(sampler, predict_fn, sample_count: int, extras: Dict[str, Any], device):
    """Collect batches of predictions and evaluate performance."""
    processed_samples = 0
    preds = []
    outputs = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        feedback = move_feedback_to_device(feedback, device)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        cur_preds, _ = predict_fn(feedback=feedback)
        preds.append(cur_preds)
        processed_samples += batch_size
    outputs = _concat(outputs, axis=0)
    preds = _concat(preds, axis=0)
    out = clrs_pytorch.evaluate(outputs, preds)
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}

def create_samplers(rng: Any,
                    train_lengths: List[int],
                    *,
                    algorithms: Optional[List[str]] = None,
                    val_lengths: Optional[List[int]] = None,
                    test_lengths: Optional[List[int]] = None,
                    train_batch_size: int = 32,
                    val_batch_size: int = 32,
                    test_batch_size: int = 32,
                    device) -> tuple:
    """Create samplers for training, validation, and testing.

    Args:
      rng: Numpy random state.
      train_lengths: List of training sample sizes.
      algorithms: List of algorithms; defaults to FLAGS.algorithms.
      val_lengths: List of validation sample sizes; defaults to maximum training length.
      test_lengths: List of test sample sizes; defaults to [-1] for the benchmark dataset.
      train_batch_size: Batch size for training.
      val_batch_size: Batch size for validation.
      test_batch_size: Batch size for testing.
      device: Device to run samplers on.
    Returns:
      Tuple containing:
        - train_samplers: List of training samplers.
        - val_samplers: List of validation samplers.
        - val_sample_counts: List of validation sample counts.
        - test_samplers: List of test samplers.
        - test_sample_counts: List of test sample counts.
        - spec_list: List of specs for each algorithm.
    """
    train_samplers = []
    val_samplers = []
    val_sample_counts = []
    test_samplers = []
    test_sample_counts = []
    spec_list = []

    algorithms = algorithms or FLAGS.algorithms
    for algo_idx, algorithm in enumerate(algorithms):
        # Run dataset pipeline on CPU.
        with device:
            if algorithm in ['naive_string_matcher', 'kmp_matcher']:
                max_length = max(train_lengths)
                if max_length > 0:
                    max_length = (max_length * 5) // 4
                train_lengths = [max_length]

            logging.info('Creating samplers for algorithm %s', algorithm)

            p = tuple(0.1 + 0.1 * i for i in range(9))
            if p and algorithm in ['articulation_points', 'bridges', 'mst_kruskal', 'bipartite_matching']:
                p = tuple(np.array(p) / 2)
            length_needle = FLAGS.length_needle
            sampler_kwargs = dict(p=p, length_needle=length_needle)
            if length_needle == 0:
                sampler_kwargs.pop('length_needle')

            common_sampler_args = dict(
                algorithm=algorithms[algo_idx],
                rng=rng,
                enforce_pred_as_input=FLAGS.enforce_pred_as_input,
                enforce_permutations=FLAGS.enforce_permutations,
                chunk_length=FLAGS.chunk_length,
            )

            train_args = dict(
                sizes=train_lengths,
                split='train',
                batch_size=train_batch_size,
                multiplier=-1,
                randomize_pos=FLAGS.random_pos,
                chunked=FLAGS.chunked_training,
                sampler_kwargs=sampler_kwargs,
                **common_sampler_args
            )
            train_sampler, _, spec = make_multi_sampler(**train_args)

            mult = clrs_pytorch.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
            val_args = dict(
                sizes=val_lengths or [np.amax(train_lengths)],
                split='val',
                batch_size=val_batch_size,
                multiplier=2 * mult,
                randomize_pos=FLAGS.random_pos,
                chunked=False,
                sampler_kwargs=sampler_kwargs,
                **common_sampler_args
            )
            val_sampler, val_samples, spec = make_multi_sampler(**val_args)

            test_args = dict(
                sizes=test_lengths or [-1],
                split='test',
                batch_size=test_batch_size,
                multiplier=2 * mult,
                randomize_pos=False,
                chunked=False,
                sampler_kwargs={},
                **common_sampler_args
            )
            test_sampler, test_samples, spec = make_multi_sampler(**test_args)

        spec_list.append(spec)
        train_samplers.append(train_sampler)
        val_samplers.append(val_sampler)
        val_sample_counts.append(val_samples)
        test_samplers.append(test_sampler)
        test_sample_counts.append(test_samples)

    return (train_samplers, val_samplers, val_sample_counts,
            test_samplers, test_sample_counts, spec_list)

def get_nb_nodes(feedback: _Feedback, is_chunked: bool) -> int:
    """Return the number of nodes from the feedback."""
    for inp in feedback.features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            return inp.data.shape[2] if is_chunked else inp.data.shape[1]
    assert False, "No valid input found."

def move_feedback_to_device(feedback, device):
    """Move all tensor-like data in feedback to the specified device."""
    def move_to_device(data):
        if isinstance(data, np.ndarray):
            return torch.tensor(data, device=device, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data

    for dp_list in [feedback.features.inputs, feedback.features.hints, feedback.outputs]:
        for dp in dp_list:
            dp.data = move_to_device(dp.data)
    return feedback

def train(model, optimizer, feedback, algo_idx: int, grad_clip_max_norm: float, device):
    """Perform a single training step."""
    feedback = move_feedback_to_device(feedback, device)
    output_preds, hint_preds = model(feedback, algo_idx)
    optimizer.zero_grad()
    lss = loss(feedback, output_preds, hint_preds, device)
    lss.backward()
    if grad_clip_max_norm != 0.0:
        clip_grad_norm_(model.parameters(), grad_clip_max_norm)
    optimizer.step()
    return lss

def loss(feedback, output_preds, hint_preds, device, decode_hints: bool = True):
    """Calculate the overall loss from output and hint losses."""
    nb_nodes = get_nb_nodes(feedback, is_chunked=False)
    lengths = feedback.features.lengths
    total_loss = 0.0

    for truth in feedback.outputs:
        loss_val = losses.output_loss(
            truth=truth,
            pred=output_preds[truth.name],
            nb_nodes=nb_nodes,
            device=device
        )
        total_loss += loss_val

    if decode_hints:
        for truth in feedback.features.hints:
            loss_val = losses.hint_loss(
                truth=truth,
                preds=[x[truth.name] for x in hint_preds],
                lengths=lengths,
                nb_nodes=nb_nodes,
                device=device
            )
            total_loss += loss_val

    return total_loss

def predict(model, feedback: _Feedback, spec, algorithm_index: int,
            return_hints: bool = False, return_all_outputs: bool = False):
    """Run prediction and post-process outputs."""
    outs, hint_preds = model(feedback, algorithm_index, repred=True,
                              return_hints=return_hints,
                              return_all_outputs=return_all_outputs)
    outs = decoders.postprocess(
        spec[algorithm_index],
        outs,
        sinkhorn_temperature=0.1,
        sinkhorn_steps=50,
        hard=True
    )
    return outs, hint_preds

def restore_model(model, checkpoint_path: str):
    """Restore model from the specified checkpoint path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

def save_model(model, output_path: str):
    """Save the model checkpoint to the specified output path."""
    checkpoint = {'model_state_dict': model.state_dict()}
    torch.save(checkpoint, output_path)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def save_results(results):
    """Save results to the performance path."""
    try:
        with open(FLAGS.performance_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info('Results successfully saved to %s', FLAGS.performance_path)
    except Exception as e:
        logging.error('Failed to save results: %s', str(e))

def restore_model_if_needed(model):
    """Restore model if the resume flag is set and a checkpoint exists."""
    if FLAGS.resume and os.path.exists(FLAGS.checkpoint_path):
        logging.info('Resuming from checkpoint: %s', FLAGS.checkpoint_path)
        restore_model(model, FLAGS.checkpoint_path)
    else:
        logging.info('No checkpoint found or resume flag is not set. Starting fresh.')


def main(unused_argv):
    # Determine hint configuration.
    if FLAGS.hint_mode == 'encoded_decoded':
        encode_hints, decode_hints = True, True
    elif FLAGS.hint_mode == 'decoded_only':
        encode_hints, decode_hints = False, True
    elif FLAGS.hint_mode == 'none':
        encode_hints, decode_hints = False, False
    else:
        raise ValueError('Hint mode must be one of {encoded_decoded, decoded_only, none}.')

    train_lengths = [int(x) for x in FLAGS.train_lengths]
    rng = np.random.RandomState(FLAGS.seed)
    set_seed(FLAGS.seed)

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else 
        'cpu'
    )

    # Create samplers.
    (train_samplers,
     val_samplers,
     val_sample_counts,
     test_samplers,
     test_sample_counts,
     spec_list) = create_samplers(
        rng=rng,
        train_lengths=train_lengths,
        algorithms=FLAGS.algorithms,
        val_lengths=[np.amax(train_lengths)],
        test_lengths=[2 * np.amax(train_lengths)],
        train_batch_size=FLAGS.batch_size,
        device=device
    )

    # Build processor-based model.
    processor_factory = clrs_pytorch.get_processor_factory(
        FLAGS.processor_type,
        use_ln=FLAGS.use_ln,
        nb_triplet_fts=FLAGS.nb_triplet_fts,
        nb_heads=FLAGS.nb_heads
    )
    model_params = dict(
        processor_factory=processor_factory,
        hidden_dim=FLAGS.hidden_size,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        encoder_init=FLAGS.encoder_init,
        use_lstm=FLAGS.use_lstm,
        learning_rate=FLAGS.learning_rate,
        grad_clip_max_norm=FLAGS.grad_clip_max_norm,
        checkpoint_path=FLAGS.checkpoint_path,
        freeze_processor=FLAGS.freeze_processor,
        dropout_prob=FLAGS.dropout_prob,
        hint_teacher_forcing=FLAGS.hint_teacher_forcing,
        hint_repred_mode=FLAGS.hint_repred_mode,
        nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
    )

    model = clrs_pytorch.models.BaselineModel(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in val_samplers],
        device=device,
        **model_params
    ).to(device)

    restore_model_if_needed(model)
    
    best_score = -1.0
    current_train_items = [0] * len(FLAGS.algorithms)
    step = 0
    next_eval = 0
    val_scores = [-99999.9] * len(FLAGS.algorithms)

    # Load previous performance results if resuming
    results = {
        "valid_accuracies": {algo: [] for algo in FLAGS.algorithms},
        "test_accuracies": {algo: [] for algo in FLAGS.algorithms}
    }
    if FLAGS.resume and os.path.exists(FLAGS.performance_path):
        try:
            with open(FLAGS.performance_path, 'r') as f:
                results = json.load(f)
            logging.info('Loaded previous results from %s', FLAGS.performance_path)
        except Exception as e:
            logging.error('Failed to load previous results: %s', str(e))
    
    # Assume all algorithms have the same number of eval steps recorded.
    if FLAGS.resume and results["valid_accuracies"][FLAGS.algorithms[0]]:
        step = len(results["valid_accuracies"][FLAGS.algorithms[0]]) * FLAGS.eval_every
        next_eval = step + FLAGS.eval_every
        logging.info("Resuming training from step %d", step)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Training loop.
    while step < FLAGS.train_steps:
        feedback_list = [next(t) for t in train_samplers]

        # Perform training step for each algorithm.
        for algo_idx, feedback in enumerate(feedback_list):
            model.train()
            cur_loss = train(model, optimizer, feedback, algo_idx,
                             FLAGS.grad_clip_max_norm, device)
            current_train_items[algo_idx] += len(feedback.features.lengths)
            logging.info('Algo %s step %i: current loss %f, total items %i.',
                         FLAGS.algorithms[algo_idx], step,
                         cur_loss, current_train_items[algo_idx])

        # Periodically evaluate model.
        if step >= next_eval:
            model.eval()
            for algo_idx in range(len(train_samplers)):
                common_extras = {
                    'examples_seen': current_train_items[algo_idx],
                    'step': step,
                    'algorithm': FLAGS.algorithms[algo_idx]
                }
                val_stats = collect_and_eval(
                    val_samplers[algo_idx],
                    functools.partial(predict, model=model, algorithm_index=algo_idx, spec=spec_list),
                    val_sample_counts[algo_idx],
                    extras=common_extras,
                    device=device
                )
                logging.info('(val) Algo %s step %d: %s',
                             FLAGS.algorithms[algo_idx], step, val_stats)
                val_scores[algo_idx] = val_stats['score']
                results["valid_accuracies"][FLAGS.algorithms[algo_idx]].append(val_stats['score'])

            next_eval += FLAGS.eval_every

            msg = (f'Best avg val score: {best_score/len(FLAGS.algorithms):.3f}, '
                   f'Current avg val score: {np.mean(val_scores):.3f}, '
                   f'Val scores: ' +
                   ', '.join([f'{algo}: {score:.3f}' for algo, score in zip(FLAGS.algorithms, val_scores)]))
            if sum(val_scores) > best_score or step == 0:
                best_score = sum(val_scores)
                logging.info('Checkpointing best model: %s', msg)
                save_model(model, FLAGS.checkpoint_path)
            else:
                logging.info('Not saving new best model: %s', msg)

            save_results(results)

        step += 1

    logging.info('Restoring best model from checkpoint...')
    restore_model(model, FLAGS.checkpoint_path)

    # Evaluate on test set.
    model.eval()
    for algo_idx in range(len(train_samplers)):
        common_extras = {
            'examples_seen': current_train_items[algo_idx],
            'step': step,
            'algorithm': FLAGS.algorithms[algo_idx]
        }
        test_stats = collect_and_eval(
            test_samplers[algo_idx],
            functools.partial(predict, model=model, algorithm_index=algo_idx, spec=spec_list),
            test_sample_counts[algo_idx],
            extras=common_extras,
            device=device
        )
        logging.info('(test) Algo %s: %s', FLAGS.algorithms[algo_idx], test_stats)
        results["test_accuracies"][FLAGS.algorithms[algo_idx]].append(test_stats['score'])


    save_results(results)
    logging.info('Training complete. Final results saved.')

if __name__ == '__main__':
    app.run(main)
