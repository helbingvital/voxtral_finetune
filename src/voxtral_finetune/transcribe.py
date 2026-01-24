import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Basic imports
from typing import Dict, Any, List
import os
import numpy as np
import evaluate
import json
import csv
import argparse
import re
import jiwer
from pathlib import Path
from collections import defaultdict
#ML related imports
from transformers import (
    AutoProcessor,
    AutoModel,
    Seq2SeqTrainer,
    IntervalStrategy,
    Seq2SeqTrainingArguments,
)
import torch
import torch.nn.functional as F
#Own code imports
from voxtral_finetune.collator import VoxtralCollatorTranscriptionTask
from voxtral_finetune.dataset import load_train_val_data, load_test_data
from voxtral_finetune.utils import get_abs_project_root_path
from voxtral_finetune.train import CustomSeq2SeqTrainer, _build_wer_fn
from voxtral_finetune.main import load_config

DEFAULT_CONFIG_FILE_NAME="config/voxtral-3B-v0.yaml"

def transcribe(cfg: Any, config_name: str, split: str="test") -> None:

    if cfg.get("dataset") is None or cfg.get("dataset").get("name") is None:
        raise ValueError("Please provide a dataset configuration in the config file under 'dataset' and 'dataset.name'.")
    
    print("Loading dataset...")
    ds={}
    ds_train, ds_eval, ds_test = None, None, None
    if split in ["train", "validation", "all"]:
        ds["train"], ds["eval"] = load_train_val_data(cfg["dataset"])
        print("Len train ds:", len(ds["train"]))
        print("Len val dataset:", len(ds["eval"]))
    if split in ["test", "all"]:
        ds["test"] = load_test_data(cfg["dataset"])
        print("Len test dataset:", len(ds["test"]))

    # optional
    # ds_train = add_transform_to_dataset(ds_train, cfg, build_augment_pipeline(cfg.get("augmentations")))
    # ds_eval = add_transform_to_dataset(ds_eval, cfg, None)#.select(range(100))
    
    fp16 = cfg.get("training", {}).get("fp16", False)
    bf16 = cfg.get("training", {}).get("bf16", True)
    if fp16 and bf16:
        raise Exception("Only select bf16 or fp16 in config file.")
    dtype = torch.bfloat16 if bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(cfg.get("model", {}).get("pretrained", ""))

    # model = AutoModel.from_pretrained(cfg.get("model", {}).get("pretrained", ""), torch_dtype=dtype)
    model = AutoModel.from_pretrained(cfg.get("inference", {}).get("model", ""), torch_dtype=dtype)

    
    
    args = Seq2SeqTrainingArguments(
        output_dir=cfg["training"].get("output_dir", os.path.join(get_abs_project_root_path(), f"weights/{config_name}/")),
        # per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        # gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        # learning_rate=cfg["training"]["learning_rate"],
        # weight_decay=cfg["training"]["weight_decay"],
        # warmup_steps=cfg["training"].get("warmup_steps", 500),
        # num_train_epochs=cfg["training"].get("num_train_epochs", 15),
        # fp16=fp16,
        bf16=bf16,
        # eval_strategy=IntervalStrategy.STEPS,
        # eval_steps=cfg["training"].get("eval_steps", 5000),
        # save_strategy = IntervalStrategy.STEPS,
        # save_steps=cfg["training"].get("save_steps", 5000),
        # save_total_limit=cfg["training"].get("save_total_limit", 1),
        remove_unused_columns=False,
        # logging_steps=cfg["training"].get("logging_steps", 25),
        # label_smoothing_factor=cfg["training"].get("label_smoothing_factor", 0),
        report_to=["tensorboard"],
        metric_for_best_model=cfg["training"].get("metric_for_best_model"),
        greater_is_better=cfg["training"].get("greater_is_better"),
        dataloader_num_workers=cfg["training"].get("dataloader_num_workers"),
        # gradient_checkpointing=False,
        # optim="adamw_bnb_8bit",
        # max_grad_norm=cfg["training"].get("max_grad_norm", None),
        predict_with_generate=cfg["inference"].get("predict_with_generate", True),
        generation_num_beams=cfg["inference"].get("num_beams", 1),
        generation_max_length=cfg["inference"].get("generation_max_length", 500)
    )
    language = cfg["inference"].get("language", None)
    collator = VoxtralCollatorTranscriptionTask(processor, 
                                                cfg.get("model", {}).get("pretrained", ""), 
                                                language=language,
                                                use_fp16=args.fp16)

    metric_file = os.path.join(os.path.dirname(__file__), "wer")
    compute_metrics = _build_wer_fn(processor, evaluate.load(metric_file))

    trainer = CustomSeq2SeqTrainer(
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        model=model,
        args=args,
        data_collator=collator,
        processor=processor,
        compute_metrics=compute_metrics,
    )
    metrics, preds, labels = {}, [], []
    output_dir = cfg["inference"].get("output_dir", os.path.join(get_abs_project_root_path(), f"runs/{config_name}/"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Saving results to ", output_dir)
    for key, dataset in ds.items():  
        print(f"Starting transcription of {key}...")      
        results = trainer.predict(dataset)
        print("Finished.")
        prediction_ids = results.predictions
        metrics[key] = results.metrics

        decoded = processor.batch_decode(prediction_ids[:, :prediction_ids.shape[1]], skip_special_tokens=True) #with language token
        label = processor.batch_decode(results.label_ids[:, :results.label_ids.shape[1]], skip_special_tokens=True) #without
        decoded = [re.sub(r'lang:[a-z]{2}', '', x) for x in decoded]
        labels+=label
        preds+=decoded        
    
    print(f"Len Labels: {len(labels)}, Len Preds: {len(preds)}")
    # if len(labels)==1:
    #     labels = labels[0]
    # elif len(labels)<4:
    #     labels = sum(labels, [])
    # if len(preds)==1:
    #     preds = preds[0]
    # elif len(labels)<4:
    #     preds = sum(preds, [])  

    with open(output_dir / f'{split}_transcriptions.csv', 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Pred", "Label"])
        writer.writerows(zip(preds, labels))

    metrics = compute_metrics_jiwer(preds, labels, output_dir, split)

    print("Results saved.")

def compute_metrics_jiwer(transcriptions: List, labels: List, result_dir: Path, split: str="test"):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    out_words = jiwer.process_words(reference=labels, hypothesis=transcriptions)
    word_alignment = jiwer.visualize_alignment(out_words)
    out_chars = jiwer.process_characters(reference=labels, hypothesis=transcriptions)
    char_alignment = jiwer.visualize_alignment(out_chars)
    word_error_metric = {"wer": out_words.wer, "mer": out_words.mer, "wil": out_words.wil, "wip": out_words.wip, \
                "hits": out_words.hits, "substitutions": out_words.substitutions, "insertions": out_words.insertions, "deletions": out_words.deletions}
    char_error_metric = {"cer": out_chars.cer,  \
                "hits": out_chars.hits, "substitutions": out_chars.substitutions, "insertions": out_chars.insertions, "deletions": out_chars.deletions}
    
    print(f"Saving metrics to {result_dir}...")
    with open(result_dir / f"{split}_alignment_words.txt", 'w') as f:
        f.write(word_alignment)
    with open(result_dir / f"{split}_alignment_chars.txt", 'w') as f:
        f.write(char_alignment)
    with open(result_dir / f"{split}_error_freq_words.txt", 'w') as f:
        f.write(jiwer.visualize_error_counts(out_words))
    with open(result_dir / f"{split}_error_freq_chars.txt", 'w') as f:
        f.write(jiwer.visualize_error_counts(out_chars))

    metric_punctuation = punctuation_error_rate(transcriptions, labels)
    metrics = {'words': word_error_metric, "chars": char_error_metric, "punctuation":metric_punctuation}

    with open(result_dir / f"{split}_metrics.json", 'w') as f:
        json.dump(metrics, f)

    return metrics

def _punctuation_error_rate(transcription: str, label: str, replacement_dict: Dict):
    """Counts the punctuation symbols specified in values of replacement_dict"""
    result={}
    total_pred=0
    total_ref = 0
    for key, value in replacement_dict.items():
        if key == '..': continue
        pred_count = transcription.count(value)
        ref_count = label.count(value)
        total_pred += pred_count
        total_ref += ref_count
        if ref_count:
            ratio = pred_count/ref_count 
        else:
            if pred_count:
                ratio = np.inf
            else:
                ratio = 1
        result[key] = {'ratio': ratio, 
                       'pred_count': pred_count, 
                       'ref_count': ref_count, 
                       'miss': pred_count<ref_count,
                       'false_positive': ref_count<pred_count}
    ratio = total_pred / total_ref if total_ref else np.inf if total_pred else 1
    result["total"] = {'ratio': ratio,
                       'pred_count': total_pred,
                       'ref_count': total_ref,
                       'miss': total_pred < total_ref,
                       'hit': total_ref < total_pred
                       }
    return result

def punctuation_error_rate(transcriptions: List, labels: List, replacement_dict=None):
    total_pred=0
    total_ref = 0
    if not replacement_dict:
        replacement_dict = {
            ' <PUNKT>': '.',
            ' <KOMMA>': ',',
            ' <DOPPELPUNKT>': ':',
            ' <KLAMMER_ZU>': ')',
            '<KLAMMER_AUF> ': '(',
            ' <NEUE_ZEILE> ': '.\n',
            ' <ABSATZ> ': '.\n\n',
            ' <NEUER_ABSATZ> ': '.\n\n',
            '..': '.',
            }
    result_list = [_punctuation_error_rate(transc, label, replacement_dict) for transc, label in zip(transcriptions, labels)]
    summary = defaultdict(lambda: {'total_pred': 0, 'total_ref': 0})

    for result in result_list:
        for symbol, metrics in result.items():
            summary[symbol]['total_pred'] += metrics['pred_count']
            summary[symbol]['total_ref'] += metrics['ref_count']

    for symbol, metric in summary.items():
        ratio = summary[symbol]['total_pred'] / summary[symbol]['total_ref'] if summary[symbol]['total_ref'] else np.inf if summary[symbol]['total_pred'] else 1
        summary[symbol]['ratio'] = ratio
    return summary

def main():
    p = argparse.ArgumentParser(description="Fine-tune Voxtral")
    p.add_argument("config_name", type=str,  nargs='?', default=None, help="Name of the YAML file in the config folder (without .yaml)")
    p.add_argument("--split", default='test', type=str, choices=['train', 'valid', 'test', 'all'])
    args = p.parse_args()

    if args.config_name is None:
        project_dir = get_abs_project_root_path()
        config_file = os.path.join(project_dir, DEFAULT_CONFIG_FILE_NAME)
    else:
        config_file = args.config_name
    
    print("Loading config...")
    cfg = load_config(config_file)
    transcribe(cfg, os.path.basename(config_file).split(".")[0], args.split)

if __name__ == "__main__":
    main()