from typing import Dict, Any, Optional, Union, Unpack
import importlib
from functools import partial

import os
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import json
import csv

from audiomentations import Compose
from transformers import (
    AutoProcessor,
    AutoModel,
    Seq2SeqTrainer,
    IntervalStrategy,
    Seq2SeqTrainingArguments,
    VoxtralForConditionalGeneration,
)
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.cache_utils import Cache
from accelerate.utils import convert_to_fp32
from voxtral_finetune.collator import VoxtralCollatorTranscriptionTask
from voxtral_finetune.dataset import load_train_val_data
from voxtral_finetune.utils import get_abs_project_root_path

from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

CUSTOM_AUGMENTATIONS = {
    "LengthAwareTimeStretch": "voxtral_finetune.augmentations",
}

def build_augment_pipeline(cfg):
    """ Builds a audiomentations pipeline from the config file. """
    transforms = []
    for t in cfg.get("transforms", []):
        cls_name = t["name"]
        if cls_name in CUSTOM_AUGMENTATIONS:
            print("Importing custom augmentation...")
            import_path = CUSTOM_AUGMENTATIONS[cls_name]
        else:
            print("Importing standard augmentation...")
            import_path = "audiomentations"
        cls = getattr(importlib.import_module(import_path), cls_name)
        transforms.append(cls(p=t.get("p", 1.0), **t.get("args", {})))
    return Compose(transforms, p=cfg.get("p", 1.0))


def dynamic_transform(batch, aug_compose_fn=None):
    """ Dynamic dataset transform, because we use audiomentations. """
    arrays = [x["array"].astype(np.float32) for x in batch["audio"]]
    sampling_rates = [x["sampling_rate"] for x in batch["audio"]]

    if aug_compose_fn is not None:
        arrays = [aug_compose_fn(samples=a, sample_rate=sr) for a, sr in zip(arrays, sampling_rates)]

    audio_objs = [{"array": a, "sampling_rate": sr} for a, sr in zip(arrays, sampling_rates)]

    return {"audio": audio_objs, "labels": batch["labels"]}


def add_transform_to_dataset(ds, cfg, aug_compose_fn=None):
    return ds.with_transform(partial(dynamic_transform, aug_compose_fn=aug_compose_fn))

class CustomVoxtralForConditionalGeneration(VoxtralForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import VoxtralForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> repo_id = "mistralai/Voxtral-Mini-3B-2507"

        >>> processor = AutoProcessor.from_pretrained(repo_id)
        >>> model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

        >>> conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(conversation)
        >>> inputs = inputs.to(device, dtype=torch.bfloat16)

        >>> outputs = model.generate(**inputs, max_new_tokens=30)
        >>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ["This audio is a humorous conversation between two friends, likely in English, where one of them is trying to figure out what the other's tattoo says."]
        ```"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_embeds = self.get_audio_embeds(input_features)
            if inputs_embeds.dtype == torch.float32:
                audio_embeds = convert_to_fp32(audio_embeds) #CHV: Match dtype of inputs_embeds
            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = input_ids == self.config.audio_token_id 
            inputs_embeds = inputs_embeds.clone() #CHV: change to work with peft model
            inputs_embeds[audio_token_mask] = audio_embeds

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return outputs

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.transcribe_token_id = 34  # denotes the [TRANSCRIBE] token, i.e. start of transcription after the language tokens (e.g. lang:de)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = model(**inputs).loss.mean().detach() if "labels" in inputs else None

        def _first_eos(seq: torch.Tensor):
            idx = (seq == self.transcribe_token_id).nonzero(as_tuple=False)
            return (idx[0, 0] + 1).item() if idx.numel() else seq.size(0)
        
        # find positions of transcription tokens + 1
        split_token_positions = [_first_eos(x) for x in inputs["input_ids"]]
        
        def _trim_leading_value(x: torch.Tensor, value: int) -> torch.Tensor:
            mask = x.ne(value)
            if not mask.any():
                return x.new_empty((0,), dtype=x.dtype)
            first = mask.nonzero(as_tuple=False)[0].item()
            return x[first:]
        
        # cut off completion to obtain the prompts
        prompt_list = [_trim_leading_value(x[:cut], 11) for (x, cut) in zip(inputs["input_ids"], split_token_positions)]

        def _pad_left_and_stack(tensor_list, pad_token_id):
            max_len = max(t.size(0) for t in tensor_list)
            padded = [F.pad(t, (max_len - t.size(0), 0), value=pad_token_id) for t in tensor_list]
            return torch.stack(padded, dim=0)

        prompt_tensor = _pad_left_and_stack(prompt_list, 11)  # TODO: if all items in the batch have audio <30, this function has no effect. It might not even have an effect if audio is longer. 
        attention_mask = (prompt_tensor != 11).long()

        with torch.no_grad():
            generated_tokens = model.generate(
                prompt_tensor,
                attention_mask=attention_mask,
                input_features = inputs["input_features"],
                max_new_tokens=422,
            )
        
        labels = inputs.get("labels")

        return (loss, generated_tokens, labels)


def train(cfg: Any, config_name: str) -> None:

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

    model = CustomVoxtralForConditionalGeneration.from_pretrained(cfg.get("model", {}).get("pretrained", ""), torch_dtype=dtype, load_in_8bit=True)
    # model = prepare_model_for_kbit_training(model, output_embedding_layer_name="proj_out")
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(r=32, 
                        lora_alpha=64, 
                        # target_modules="all-linear", 
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05, 
                        bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

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
        report_to=["tensorboard", "wandb"],
        metric_for_best_model=cfg["training"].get("metric_for_best_model"),
        greater_is_better=cfg["training"].get("greater_is_better"),
        dataloader_num_workers=cfg["training"].get("dataloader_num_workers"),
        gradient_checkpointing=False,
        optim="adamw_bnb_8bit",
        max_grad_norm=cfg["training"].get("max_grad_norm", None),
        predict_with_generate=False,
        label_names=["labels"],
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
    
    print("Starting training...")
    trainer.train()
    
    
def _build_wer_fn(processor, wer_metric):
  import re
  def _compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
      labels = pred.label_ids.copy()
      labels[labels == -100] = processor.tokenizer.pad_token_id
      labels_text = processor.tokenizer.batch_decode(labels,            skip_special_tokens=True)
      preds_text  = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

      # Voxtral has language tokens as non-special tokens? therefore, we need to manually remove them. TODO: move this upstream (prediction_step fn)
      preds_text = [re.sub(r'lang:[a-z]{2}', '', x) for x in preds_text]
      pairs = [(p.strip(), l.strip()) for p, l in zip(preds_text, labels_text) if p.strip() and l.strip()]
      
      for pair in pairs:
          break
      if not pairs:
          wer_overall = 0.0
      else:
          wer_overall = 100.0 * wer_metric.compute(
              predictions=[p for p, _ in pairs],
              references=[l for _, l in pairs]
          )
      return {"wer": float(wer_overall)}
  return _compute_metrics

