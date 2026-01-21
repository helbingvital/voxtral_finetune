from typing import Dict, Any

import os
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import json
import csv

from transformers import (
    AutoProcessor,
    AutoModel,
    Seq2SeqTrainer,
    IntervalStrategy,
    Seq2SeqTrainingArguments,
)

from voxtral_finetune.collator import VoxtralCollatorTranscriptionTask
from voxtral_finetune.dataset import load_train_val_data
from voxtral_finetune.utils import get_abs_project_root_path
from voxtral_finetune.train import CustomSeq2SeqTrainer, _build_wer_fn

def transcribe(cfg: Any, config_name: str) -> None:

    if cfg.get("dataset") is None or cfg.get("dataset").get("name") is None:
        raise ValueError("Please provide a dataset configuration in the config file under 'dataset' and 'dataset.name'.")
    
    print("Loading dataset...")
    ds_train, ds_eval = load_train_val_data(cfg["dataset"])

    # optional
    # ds_train = add_transform_to_dataset(ds_train, cfg, build_augment_pipeline(cfg.get("augmentations")))
    # ds_eval = add_transform_to_dataset(ds_eval, cfg, None)#.select(range(100))
   
    print("Len train ds:", len(ds_train))
    print("Len val dataset:", len(ds_eval))
    
    fp16 = cfg.get("training", {}).get("fp16", False)
    bf16 = cfg.get("training", {}).get("bf16", True)
    if fp16 and bf16:
        raise Exception("Only select bf16 or fp16 in config file.")
    dtype = torch.bfloat16 if bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(cfg.get("model", {}).get("pretrained", ""))

    model = AutoModel.from_pretrained(cfg.get("model", {}).get("pretrained", ""), torch_dtype=dtype)

    
    
    args = Seq2SeqTrainingArguments(
        output_dir=cfg["training"].get("output_dir", os.path.join(get_abs_project_root_path(), f"weights/{config_name}/")),
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"].get("warmup_steps", 500),
        num_train_epochs=cfg["training"].get("num_train_epochs", 15),
        # fp16=fp16,
        bf16=bf16,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=cfg["training"].get("eval_steps", 5000),
        save_strategy = IntervalStrategy.STEPS,
        save_steps=cfg["training"].get("save_steps", 5000),
        save_total_limit=cfg["training"].get("save_total_limit", 1),
        remove_unused_columns=False,
        logging_steps=cfg["training"].get("logging_steps", 25),
        label_smoothing_factor=cfg["training"].get("label_smoothing_factor", 0),
        report_to=["tensorboard"],
        metric_for_best_model=cfg["training"].get("metric_for_best_model"),
        greater_is_better=cfg["training"].get("greater_is_better"),
        dataloader_num_workers=cfg["training"].get("dataloader_num_workers"),
        gradient_checkpointing=False,
        optim="adamw_bnb_8bit",
        max_grad_norm=cfg["training"].get("max_grad_norm", None),
        predict_with_generate=False,
    )
    collator = VoxtralCollatorTranscriptionTask(processor, 
                                                cfg.get("model", {}).get("pretrained", ""), 
                                                language="de",
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
    
    
    print("Starting transcription...")
    results = trainer.predict(ds_eval)
    print("Finished.")
    prediction_ids = results.predictions
    metrics = results.metrics
    with open("debug_trainer_predict.json", 'w') as f:
        json.dump(metrics, f)
    decoded = processor.batch_decode(prediction_ids[:, :prediction_ids.shape[1]], skip_special_tokens=True) #with language token
    labels = processor.batch_decode(results.label_ids[:, :results.label_ids.shape[1]], skip_special_tokens=True) #without
    with open('./playground/transcriptions.csv', 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Pred", "Label"])
        writer. writerows(zip(decoded, labels))

    print("Results saved.")
