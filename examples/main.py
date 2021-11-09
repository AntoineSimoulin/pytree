import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    Trainer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
# from utils_qa import postprocess_qa_predictions

from pytree import (
    NaryConfig, 
    NaryTree,
    ChildSumConfig,
    ChildSumTree,
    GloveTokenizer,
    Similarity,
    SimilarityConfig
)
from pytree.data import prepare_input_from_constituency_tree, prepare_input_from_dependency_tree
from pytree.data.utils import build_tree_ids_n_ary

from supar import Parser
import torch
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")  # 4.12.0.dev0

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

class SickTrainer(Trainer):
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.025, weight_decay=self.args.weight_decay)
    
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     self.optimizer = OSS(
        #         params=optimizer_grouped_parameters,
        #         optim=optimizer_cls,
        #         **optimizer_kwargs,
        #     )
        # else:
        #     self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # if is_sagemaker_mp_enabled():
        #     self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


con = Parser.load('crf-con-en')
glove_tokenizer = GloveTokenizer(glove_file_path='/data/asimouli/GLOVE/glove.6B.300d.txt', vocab_size=10000)

config = NaryConfig()
encoder = NaryTree(config)
encoder.embeddings.load_pretrained_embeddings(
    torch.tensor(glove_tokenizer.embeddings_arr))    
config_similarity = SimilarityConfig()
model = Similarity(encoder, config_similarity)

raw_datasets = load_dataset('sick')
column_names = raw_datasets["train"].column_names

def map_label_to_target(label, num_classes):
    target = [0] * num_classes  # torch.zeros(1, num_classes, dtype=torch.float)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[floor - 1] = 1
    else:
        target[floor - 1] = ceil - label
        target[ceil - 1] = label - floor
    return target

def prepare_train_features(examples):
    examples['input_ids_A'] = []
    examples['input_ids_B'] = []
    examples['head_idx_A'] = []
    examples['head_idx_B'] = []
    examples['labels'] = []
    
    for sent_A in examples['sentence_A']:
        con_tree_A = str(con.predict(sent_A.split(), verbose=False)[0])
        input_ids_A, head_idx_A = prepare_input_from_constituency_tree(con_tree_A)
        input_ids_A = glove_tokenizer.convert_tokens_to_ids(input_ids_A)
        examples['input_ids_A'].append(input_ids_A)
        examples['head_idx_A'].append(head_idx_A)
    
    for sent_B in examples['sentence_B']:
        con_tree_B = str(con.predict(sent_B.split(), verbose=False)[0])
        input_ids_B, head_idx_B = prepare_input_from_constituency_tree(con_tree_B)
        input_ids_B = glove_tokenizer.convert_tokens_to_ids(input_ids_B)
        examples['input_ids_B'].append(input_ids_B)
        examples['head_idx_B'].append(head_idx_B)

    for rel_score in examples['relatedness_score']:
        examples['labels'].append(map_label_to_target(rel_score, 5))

    return examples

training_args = TrainingArguments(
    learning_rate=0.025, 
    per_device_train_batch_size=25, 
    num_train_epochs=20, 
    weight_decay=1e-4, 
    lr_scheduler_type='constant', 
    output_dir="/home/asimouli/PhD/PyTree/pytree_remote/model", 
    do_train=True, 
    do_eval=True,
    remove_unused_columns=False)

train_examples = raw_datasets["train"]
with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=None,
        remove_columns=None,
        load_from_cache_file=True,
        desc="Running parser on train dataset",
    )

# # Validation preprocessing

eval_examples = raw_datasets["validation"]
eval_dataset = eval_examples.map(
    prepare_train_features,
    batched=True,
    num_proc=None,
    remove_columns=None,  # column_names,
    desc="Running parser on validation dataset",
)

def data_collator_with_padding(features, pad_ids=0, columns=None):
    batch = {}
    first = features[0]
    if columns is None:
        columns = ["head_idx_A", "head_idx_B", "input_ids_A", "input_ids_B"]
    feature_max_len = {k: max([len(f[k]) for f in features]) for k in first.keys() if k in columns or len(columns) == 0}
    for k, v in first.items():
        if k in columns or len(columns) == 0:
            feature_padded = [list([int(ff) for ff in f[k]]) + [0] * (feature_max_len[k] - len(f[k])) for f in features]
            batch[k] = feature_padded  # [f[k] for f in features]
    tree_ids_A, tree_ids_r_A, tree_ids_l_A = build_tree_ids_n_ary(batch['head_idx_A'])
    tree_ids_B, tree_ids_r_B, tree_ids_l_B = build_tree_ids_n_ary(batch['head_idx_B'])
    batch['input_ids_A'] = torch.tensor(batch['input_ids_A'])
    batch['input_ids_B'] = torch.tensor(batch['input_ids_B'])
    batch['tree_ids_A'] = torch.tensor(tree_ids_A)
    batch['tree_ids_B'] = torch.tensor(tree_ids_B)
    batch['tree_ids_r_A'] = torch.tensor(tree_ids_r_A)
    batch['tree_ids_r_B'] = torch.tensor(tree_ids_r_B)
    batch['tree_ids_l_A'] = torch.tensor(tree_ids_l_A)
    batch['tree_ids_l_B'] = torch.tensor(tree_ids_l_B)
    batch['labels'] = torch.tensor([f['labels'] for f in features])
    return batch

data_collator = data_collator_with_padding

def compute_metrics(eval_prediction):
    prediction = np.matmul(np.exp(eval_prediction.predictions), np.arange(1, 5 + 1))
    target = np.matmul(eval_prediction.label_ids, np.arange(1, 5 + 1))
    results_relatedness = {
        'pearson': pearsonr(prediction, target)[0] * 100,
        'spearman': spearmanr(prediction, target)[0] * 100,
        'mse': mean_squared_error(prediction, target) * 100
    }
    return results_relatedness
    
trainer = SickTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=("Adagrad", None),
)

# Training

train_result = trainer.train(resume_from_checkpoint=None)
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


logger.info("*** Evaluate ***")
metrics = trainer.evaluate()

max_eval_samples = len(eval_dataset)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

