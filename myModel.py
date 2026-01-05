from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import pandas as pd

from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import tqdm
from configarg import TrainingArgs, ModelArgs
from dataclasses import dataclass

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

import logging
import numpy as np
import os
import random
from submodels import *

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
import transformers
import csv


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class MyDataset(data.Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,):

        super(MyDataset, self).__init__()

        with open(data_path, 'r') as f:
            data = list(csv.reader(f))[1:]
        logging.warning("--Perform single sequence classification--")
        seqs = [d[0] for d in data]
        labels = [int(d[1]) for d in data]

        output = tokenizer(
            seqs,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.labels = labels
        self.attention_mask = output["attention_mask"]
        self.num_labels = len(set(labels))

        base2id = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
        seq_len = len(seqs[0])
        base2ids = torch.zeros((len(seqs), seq_len), dtype=torch.long)
        for i, seq in enumerate(seqs):
            seq = seq.strip().upper()
            if len(seq) != seq_len:
                raise ValueError(
                    f"Sequence length mismatch in {data_path}: "
                    f"expect {seq_len}, got {len(seq)}"
                )
            for j, ch in enumerate(seq):
                base2ids[i, j] = base2id.get(ch, 0)  # Unknown ch to A
        self.numseqs = base2ids  # (N, L_base)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], numseqs=self.numseqs[idx])

@dataclass
class DataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch):
        input_ids, labels, numseqs = tuple([instance[key] for instance in batch] for key in ("input_ids", "labels", "numseqs"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.Tensor(labels).long()
        numseqs = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in numseqs]).long()

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
            numseqs=numseqs,
        )


def calculate_metric(preds_prob, labels):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_preds_prob = preds_prob[valid_mask][:, 1]
    valid_labels = labels[valid_mask]

    preds = (valid_preds_prob >= 0.5).astype(int)
    labels = np.asarray(valid_labels)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    pre = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    sp = recall_score(labels, preds, pos_label=0)
    mcc = matthews_corrcoef(labels, preds)

    auc = float("nan")
    if valid_preds_prob is not None:
        try:
            auc = roc_auc_score(labels, valid_preds_prob)
        except ValueError:
            pass

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "specificity": sp,
        "matthews_correlation": mcc,
        "roc_auc": auc, }


def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        print(12345)
        logits = logits[0]

    return torch.softmax(logits, dim=-1)

def compute_metrics(eval_pred):
    preds_prob, labels = eval_pred   # 注意：这里的 y_pred 已是“预处理后的 predictions”
    return calculate_metric(preds_prob, labels)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, model_args: ModelArgs).__init__()

        self.bert = AutoModel.from_pretrained(model_args.bert_name_or_path, trust_remote_code=True)

        self.onehot = ONEHOT()
        self.eiip = EIIP()
        self.ncp = NCP()
        self.enac = ENAC()

        self.lstm1 = nn.LSTM(
            input_size=768,
            hidden_size=model_args.lstm1_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=model_args.lstm_dropout,
        )
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=model_args.lstm2_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=model_args.lstm_dropout,
        )

        self.dropout1 = nn.Dropout(model_args.dropout)
        self.dropout2 = nn.Dropout(model_args.dropout)

        self.cam_pick_idx = model_args.cam_pick_idx
        self.ffm_pick_idx = model_args.ffm_pick_idx

        conc_dim = 2 * (model_args.lstm1_hidden + model_args.lstm2_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(conc_dim , 128),
            nn.ReLU(),
            nn.Dropout(model_args.dropout),
            nn.Linear(128, 2),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_types_ids: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                head_mask=None,
                inputs_embeds=None,
                labels: torch.Tensor =None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                numseqs = None,
    ):
        return_dict = return_dict if return_dict is not None else True

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_types_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            )
        hidden_states = outputs[0]
        idx1 = self.cam_pick_idx
        lstm1_out, _ = self.lstm1(hidden_states)
        x1 = lstm1_out[:, idx1, :]   #do shifting to align to the center word
        x1 = self.dropout1(x1)

        onehot = self.onehot(numseqs)
        ncp = self.ncp(numseqs)
        eiip = self.eiip(numseqs)
        enac = self.enac(numseqs)

        fused_emb = torch.cat([onehot, ncp, eiip, enac], dim=-1)
        fused_emb = self.dropout2(fused_emb)
        lstm2_out, _ = self.lstm2(fused_emb)
        idx2 = self.ffm_pick_idx
        x2 = lstm2_out[:, idx2, :]
        x2 = self.dropout2(x2)

        x = torch.cat((x1, x2), dim=-1)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
        }

def train():
    parser = transformers.HfArgumentParser((ModelArgs, TrainingArgs))
    model_args, training_args = parser.parse_args_into_dataclasses()

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"{training_args.run_name}_{time_str}"
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("dnabert2/DNABERT-2-117M",
                                              model_max_length = training_args.model_max_length,
                                              cache_dir=training_args.cache_dir,
                                              padding_side = "right",
                                              use_fast=True,
                                              trust_remote_code=True,)

    train_dataset = MyDataset(tokenizer=tokenizer,data_path="data/train.csv")
    val_dataset = MyDataset(tokenizer=tokenizer,data_path="data/dev.csv")
    test_dataset = MyDataset(tokenizer=tokenizer,data_path="data/test.csv")
    data_collator = DataCollator(tokenizer=tokenizer)

    model = MyModel(model_args)

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACT",
            inference_mode=False,
        )
        model.bert = get_peft_model(model.bert, lora_config)
        model.bert.print_trainable_parameters()

    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        print("best_model_checkpoint =", trainer.state.best_model_checkpoint)
        print("best_metric =", trainer.state.best_metric)

        trainer.save_state()
        # 如果 load_best_model_at_end=True，这里的 trainer.model 就是当前最优模型
        best_output_dir = os.path.join(
            training_args.output_dir,
            "best_model",
            run_tag,  # 带时间戳，避免覆盖
        )
        os.makedirs(best_output_dir, exist_ok=True)

        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(best_output_dir, state_dict=cpu_state_dict)

    # get the evaluation results
    if training_args.eval_and_save_results:
        print("testing model")
        results_path = os.path.join(training_args.output_dir, "results", run_tag)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    train()


















