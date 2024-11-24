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

"""The CLRS Algorithmic Reasoning Benchmark."""

from clrs_pytorch import models

from clrs_pytorch._src import algorithms

from clrs_pytorch._src import decoders
from clrs_pytorch._src import processors
from clrs_pytorch._src import specs

from clrs_pytorch._src.dataset import chunkify
from clrs_pytorch._src.dataset import CLRSDataset
from clrs_pytorch._src.dataset import create_chunked_dataset
from clrs_pytorch._src.dataset import create_dataset
from clrs_pytorch._src.dataset import get_clrs_folder
from clrs_pytorch._src.dataset import get_dataset_gcp_url

from clrs_pytorch._src.evaluation import evaluate
from clrs_pytorch._src.evaluation import evaluate_hints

from clrs_pytorch._src.model import Model

from clrs_pytorch._src.probing import DataPoint
from clrs_pytorch._src.probing import predecessor_to_cyclic_predecessor_and_first

from clrs_pytorch._src.processors import get_processor_factory

from clrs_pytorch._src.samplers import build_sampler
from clrs_pytorch._src.samplers import CLRS30
from clrs_pytorch._src.samplers import Features
from clrs_pytorch._src.samplers import Feedback
from clrs_pytorch._src.samplers import process_permutations
from clrs_pytorch._src.samplers import process_pred_as_input
from clrs_pytorch._src.samplers import process_random_pos
from clrs_pytorch._src.samplers import Sampler
from clrs_pytorch._src.samplers import Trajectory

from clrs_pytorch._src.specs import ALGO_IDX_INPUT_NAME
from clrs_pytorch._src.specs import CLRS_30_ALGS_SETTINGS
from clrs_pytorch._src.specs import Location
from clrs_pytorch._src.specs import OutputClass
from clrs_pytorch._src.specs import Spec
from clrs_pytorch._src.specs import SPECS
from clrs_pytorch._src.specs import Stage
from clrs_pytorch._src.specs import Type

__version__ = "2.0.1"

__all__ = (
    "ALGO_IDX_INPUT_NAME",
    "build_sampler",
    "chunkify",
    "CLRS30",
    "CLRS_30_ALGS_SETTINGS",
    "create_chunked_dataset",
    "create_dataset",
    "clrs_text",
    "get_clrs_folder",
    "get_dataset_gcp_url",
    "get_processor_factory",
    "DataPoint",
    "predecessor_to_cyclic_predecessor_and_first",
    "process_permutations",
    "process_pred_as_input",
    "process_random_pos",
    "specs",
    "evaluate",
    "evaluate_hints",
    "Features",
    "Feedback",
    "Location",
    "Model",
    "Sampler",
    "Spec",
    "SPECS",
    "Stage",
    "Trajectory",
    "Type",
)
