# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
import torch
from dataclasses import dataclass, field, asdict
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
import time

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask

import models_ner
import thop

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {"SimpleClassifier": models_ner.SimpleClassifier,
                    "SimpleCNN": models_ner.SimpleCNN,
                    "SimpleCNNSoftmax": models_ner.SimpleCNNSoftmax,
                    "MultipleWindowCNN": models_ner.MultipleWindowCNN,
                    "MultipleWindowCNN2": models_ner.MultipleWindowCNN2,
                    "WindowSequenceModel": models_ner.WindowSequenceModel,
                    "WindowSequenceModel128": models_ner.WindowSequenceModel128,
                    "WindowSequenceModel128AllKD": models_ner.WindowSequenceModel128AllKD,
                    "WindowSequenceModelBertEmbeddingsFrozen": models_ner.WindowSequenceModelBertEmbeddingsFrozen,
                    "SimpleLSTM": models_ner.SimpleLSTM,
                    "SimpleLSTM128": models_ner.SimpleLSTM128,
                    "SimpleLSTM128AllKD": models_ner.SimpleLSTM128AllKD,
                    "SimpleLSTM128Dropout02": models_ner.SimpleLSTM128Dropout02,
                    "SimpleLSTM128Depth2": models_ner.SimpleLSTM128Depth2,
                    "SimpleLSTM128Depth2Dropout02": models_ner.SimpleLSTM128Depth2Dropout02,
                    "SimpleLSTM128Depth3Dropout02": models_ner.SimpleLSTM128Depth3Dropout02,
                    "SimpleLSTM128BertEmbeddingsFrozen": models_ner.SimpleLSTM128BertEmbeddingsFrozen,
                    "SimpleLSTM256": models_ner.SimpleLSTM256,
                    "SimpleLSTM256Dropout02": models_ner.SimpleLSTM256Dropout02,
                    "SimpleLSTM256Depth2Dropout02": models_ner.SimpleLSTM256Depth2Dropout02,
                    "SimpleLSTM256Depth2Dropout02RNNDropout02": models_ner.SimpleLSTM256Depth2Dropout02RNNDropout02,
                    "SimpleLSTM256Depth2Dropout05RNNDropout05": models_ner.SimpleLSTM256Depth2Dropout05RNNDropout05,
                    "SimpleLSTM256Depth3Dropout02": models_ner.SimpleLSTM256Depth3Dropout02,
                    "SimpleLSTM256BertEmbeddingsFrozen": models_ner.SimpleLSTM256BertEmbeddingsFrozen,
                    "SimpleLSTM512": models_ner.SimpleLSTM512}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_class: str = field(
        metadata={"help": "The class of the desired model. If 'BERT' you must also provide a 'model_name_or_path'"},
    )
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to standard BERT pretrained model or model identifier from huggingface.co/models"}
    )
    custom_model_state_dict_path: Optional[str] = field(
        default=None, metadata={"help": "Path to custom pretrained model state dict"}
    )
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to standard BERT pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    kd_param: Optional[float] = field(
        default=0,
        metadata={
            "help": "The coefficient for knowledge distillation training. Zero means no distillation training."}
    )
    loss_fct_kd: Optional[str] = field(
        default=None,
        metadata={
            "help": "The loss function to use for knowledge distillation training."
        }
    )
    bert_embeddings_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the bert embedding weights tensor if pre-trained embeddings are to be used."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    nbr_train_samples: int = field(
        default=-1,
        metadata={
            "help": "The number of train samples to produce logits for." 
        },
    )
    
def computeTime(model, inputs, device='cuda'):
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(**inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    return np.mean(time_spent)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    elif not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    config.max_seq_length = data_args.max_seq_length
    config.pad_token_id = tokenizer.pad_token_id
    config.device = training_args.device
    
    # pass BERT embeddings if should be loaded to model
    config.bert_embeddings = None
    if model_args.bert_embeddings_path:
        config.bert_embeddings = torch.load(model_args.bert_embeddings_path)
    
    config.kd_param = model_args.kd_param
    config.loss_fct_kd = None
    config.teacher_model = None

    if model_args.model_class=="BERT":
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        model_type = config.model_type
    else:
        model = MODEL_CLASS_DICT[model_args.model_class](config)
        if model_args.custom_model_state_dict_path:
            model.load_state_dict(torch.load(model_args.custom_model_state_dict_path))
        model.to(training_args.device)
        model_type = "custom"

    num_model_params = sum(p.numel() for p in model.parameters())
    
    # acquire data for macs measurement at inference
    test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

    # create test data
    model_test_input = asdict(test_dataset[0])
    inputs_dict = {}
    for key, val in model_test_input.items():
        val_tensor = torch.LongTensor(val).unsqueeze(0)
        if key == "label_ids":
            inputs_dict["labels"] = val_tensor
        else:
            inputs_dict[key] = val_tensor
    
    def count_embedding_macs_training(embedding_dim, max_seq_length, batch_size):
        # embedding_dim, one update (2?), per token in sequence
        total_ops = embedding_dim

        total_ops *= max_seq_length
        total_ops *= batch_size

        return int(total_ops) 
    
    # calculate model number of MACs at inference    
    if not model_args.model_class == "BERT":
        num_embedding_macs = count_embedding_macs_training(model.embedding.weight.shape[1], data_args.max_seq_length, 1)
    else:
        logger.info("Measuring parameters for BERT model. Will ignore embedding calculations.")
        num_embedding_macs = 0
        
    thop_num_model_macs, thop_num_model_params = thop.profile(model, inputs=inputs_dict)
    
    # calculate model inference time
    model_mean_100_inference_time = computeTime(model, inputs=inputs_dict, device='cpu')
    
    output_model_info_file = os.path.join(training_args.output_dir, "model_info.txt")
    
    with open(output_model_info_file, 'w') as f:
        f.write("Model class: \t %s" %(model_args.model_class))
        f.write('\n')
        f.write("Number of model parameters: \t %d" %(num_model_params))
        f.write('\n')
        f.write("Number of model parameters (thop): \t %d" %(thop_num_model_params))
        f.write('\n')
        f.write("Number of MACs necessary for one example forward pass: \t %d" %(thop_num_model_macs))
        f.write('\n')
        f.write("Number of MACs necessary for one example embedding training pass: \t %d" %(num_embedding_macs))
        f.write('\n')
        f.write("Time necessary for one example forward pass (average of 100 runs): \t %.3f s" %(model_mean_100_inference_time))
        f.write('\n')
    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
