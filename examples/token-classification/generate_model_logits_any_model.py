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
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

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

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {"SimpleClassifier": models_ner.SimpleClassifier,
                    "SimpleCNN": models_ner.SimpleCNN,
                    "SimpleCNNSoftmax": models_ner.SimpleCNNSoftmax,
                    "MultipleWindowCNN": models_ner.MultipleWindowCNN,
                    "MultipleWindowCNN2": models_ner.MultipleWindowCNN2,
                    "WindowSequenceModel": models_ner.WindowSequenceModel,
                    "WindowSequenceModel128": models_ner.WindowSequenceModel128,
                    "WindowSequenceModel128AllKD": models_ner.WindowSequenceModel128AllKD,
                    "SimpleLSTM": models_ner.SimpleLSTM,
                    "SimpleLSTM128": models_ner.SimpleLSTM128,
                    "SimpleLSTM128Depth2": models_ner.SimpleLSTM128Depth2}

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

def save_logits_to_file(file_path, logits):
    with open(file_path, 'w') as f:
        for batch_predictions in logits:
            for word_predictions in batch_predictions:
                f.write(' '.join([str(word_prediction) for word_prediction in word_predictions]))
                f.write('\n')
            f.write('\n')
    return

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

    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    train_dataset = TokenClassificationDataset(
        token_classification_task=token_classification_task,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.train,
    )

    if data_args.nbr_train_samples == -1:
        predictions, label_ids, metrics = trainer.predict(train_dataset)
    else:
        predictions, label_ids, metrics = trainer.predict(train_dataset[:data_args.nbr_train_samples])
    
    # save raw logits predictions
    output_train_logits_file = os.path.join(training_args.output_dir, "train_logits.txt")
    save_logits_to_file(output_train_logits_file, predictions)
            
    # save with NER tags
    preds_list, _ = align_predictions(predictions, label_ids)

    output_train_results_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_results_file, "w") as writer:
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Save train predictions
    output_train_predictions_file = os.path.join(training_args.output_dir, "train_predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_train_predictions_file, "w") as writer:
            with open(os.path.join(data_args.data_dir, "train.txt"), "r") as f:
                token_classification_task.write_predictions_to_file(writer, f, preds_list)

    # Save word piece predictions
    output_word_piece_predictions_file = os.path.join(training_args.output_dir, "train_word_piece_predictions.txt")
    assert len(train_dataset) == len(predictions)
    with open(output_word_piece_predictions_file, "w") as writer:
        writer.write("word_piece label_id prediction \n")
        for example_index, example in enumerate(train_dataset.features):
            for index, word_piece in enumerate(tokenizer.convert_ids_to_tokens(example.input_ids)):
                predicted_label_id = np.argmax(predictions[example_index][index])
                writer.write(word_piece + " " + str(example.label_ids[index]) + " " + str(predicted_label_id) + "\n")
            writer.write("\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
