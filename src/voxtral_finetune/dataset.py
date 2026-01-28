from typing import Any, Dict, Union
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datasets import Dataset, load_dataset, Audio

from voxtral_finetune.normalizer import CustomNormalizer


def _validate_dataset_split(dataset_split, split_name: str):
    """Validate that a dataset split is loaded correctly."""
    if dataset_split is None:
        raise ValueError(f"{split_name} split is None")
    
    if len(dataset_split) == 0:
        raise ValueError(f"{split_name} split is empty")
    
    required_columns = {"audio", "labels"}
    missing_columns = required_columns - set(dataset_split.column_names)
    if missing_columns:
        raise ValueError(f"{split_name} split missing required columns: {missing_columns}. Available columns: {dataset_split.column_names}. Maybe renaming is enough?")
    #on macos: export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH" because ffmpeg libavutil is not inside anaconda, fix path
    try:
        first_sample = dataset_split[0]
        
        if "audio" not in first_sample or first_sample["audio"] is None:
            raise ValueError(f"{split_name} split has invalid audio data")
        
        if "labels" not in first_sample or not isinstance(first_sample["labels"], str):
            raise ValueError(f"{split_name} split has invalid labels data")
            
    except Exception as e:
        raise ValueError(f"Error validating {split_name} split: {str(e)}")


def _load_librispeech_dataset_clean():
    train = load_dataset("openslr/librispeech_asr", split="train.clean.100")
    
    # ---- load & preprocess train ----
    train = train.rename_column("text", "labels")
    train = train.map(lambda x: {"labels": x["labels"].lower()})
    train = train.cast_column("audio", Audio(sampling_rate=16_000))

    # ---- load & preprocess val ----
    validation = load_dataset("openslr/librispeech_asr", split="validation.clean")
    validation = validation.rename_column("text", "labels")
    validation = validation.map(lambda x: {"labels": x["labels"].lower()})
    validation = validation.cast_column("audio", Audio(sampling_rate=16_000))

    def __get_audio_length(example):
        return {"audio_length": len(example["audio"]["array"]) / example["audio"]["sampling_rate"]}

    validation = validation.map(__get_audio_length)  # only validation has audio > 30 seconds
    validation = validation.filter(lambda example: example["audio_length"] <= 30)

    _validate_dataset_split(train, "train")
    _validate_dataset_split(validation, "validation")

    return train, validation

def _load_multimed_german():
    dataset = load_dataset("leduckhai/MultiMed", "German")
    
    
    # ---- load & preprocess train ----
    train = dataset["train"]
    train = train.rename_column("text", "labels")
    train = train.cast_column("audio", Audio(sampling_rate=16_000))
    train = train.filter(lambda example: example["duration"] <= 30)
    
    # ---- load & preprocess val ----
    validation = dataset["eval"]
    validation = validation.rename_column("text", "labels")
    validation = validation.cast_column("audio", Audio(sampling_rate=16_000))
    validation = validation.filter(lambda example: example["duration"] <= 30)

    _validate_dataset_split(train, "train")
    _validate_dataset_split(validation, "validation")

    return train, validation


def _convert_radmed_uka_split_csv(split_csv="./splits/medical_split_20251113.csv", 
                                  new_split_csv_directory="./splits/medical_split_20251113/",
                                  reapply_normalizer=False):
    '''Cleanup split.csv files that were created for whisper finetune to be compatible with load_dataset'''
    new_split_csv_directory = Path(new_split_csv_directory)
    new_split_csv_directory.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(split_csv)
    df = df[~df["do_not_use"]].set_index("uid")
    drop_columns = ["Unnamed: 0", "base_id", "sub_id", "do_not_use", "main_reason", "reason"]
    drop_columns = list(set(drop_columns) & set(df.columns))
    df.drop(columns=drop_columns, inplace=True,)

    if reapply_normalizer or "text_norm" not in df.columns:
        normalizer = CustomNormalizer()
        df["text_norm"]=df["text"].apply(lambda x: normalizer(x))
    df.rename(columns={"text": "text_org", "text_norm": "text_norm_org"}, inplace=True)
    df["text"] = df["text_org"].apply(lambda x: _apply_replacement_dict(x))
    df["text_norm"] = df["text_norm_org"].apply(lambda x: _apply_replacement_dict(x))

    df_train = df[df["split"]=="train"].drop(columns="split").to_csv(new_split_csv_directory / "train.csv")
    df_eval = df[df["split"]=="valid"].drop(columns="split").to_csv(new_split_csv_directory / "valid.csv")
    df_test = df[df["split"]=="test"].drop(columns="split").to_csv(new_split_csv_directory / "test.csv")

def _apply_replacement_dict(text, replace_dict= 
                            {' <PUNKT>': '.', ' <KOMMA>': ',', ' <DOPPELPUNKT>': ':', ' <KLAMMER_ZU>': ')','<KLAMMER_AUF> ': '(',' <NEUE_ZEILE> ': '.\n',' <ABSATZ> ': '.\n\n',' <NEUER_ABSATZ> ': '.\n\n','..': '.',}):
    for old, new in replace_dict.items():
        text = text.replace(old, new)
    return text

def _load_radmed_uka(datafiles, root_path="/Users/helbing/", use_normalized_label=True, apply_replacement_dict=True, splits=["train", "validation"]):
    
    dataset = load_dataset("csv", data_files=datafiles)
    
    def prepend_root(batch):
        batch["audio"] = [os.path.join(root_path, path) for path in batch["audio"]]
        return batch

    dataset = dataset.map(prepend_root, batched=True)

    if use_normalized_label:
        label_column = "text_norm" if apply_replacement_dict else "text_norm_org"
    else:
        label_column = "text" if apply_replacement_dict else "text_org"
        
    ds = []
    if "train" in splits:
        train = dataset['train']
        train = train.rename_column(label_column, "labels") if use_normalized_label else train.rename_column("text", "labels")
        train = train.cast_column("audio", Audio(sampling_rate=16_000))
        # train = train.filter(lambda example: example["duration"] <= 30) #redundant, in clean split.csv they are already filtered
        _validate_dataset_split(train, "train")
        ds.append(train)
    
    if "validation" in splits:
        validation = dataset["eval"]
        validation = validation.rename_column(label_column, "labels") if use_normalized_label else train.rename_column("text", "labels")
        validation = validation.cast_column("audio", Audio(sampling_rate=16_000))
        # validation = validation.filter(lambda example: example["duration"] <= 30)
        _validate_dataset_split(validation, "validation")
        ds.append(validation)

    if "test" in splits:
        test = dataset['test']
        test = test.rename_column("text_norm", "labels") if use_normalized_label else test.rename_column("text", "labels")
        test = test.cast_column("audio", Audio(sampling_rate=16_000))
        # train = train.filter(lambda example: example["duration"] <= 30) #redundant, in clean split.csv they are already filtered
        _validate_dataset_split(test, "test")
        ds.append(test)

    return ds

# def _load_radmed_uka_test(datafiles=None, root_path="/Users/helbing/", use_normalized_label=True, apply_replacement_dict=True):
#     if not datafiles:
#         datafiles = {"test":"splits/medical_split_20251113/test.csv",}
#     dataset = load_dataset("csv", data_files=datafiles)
    
#     def prepend_root(batch):
#         batch["audio"] = [os.path.join(root_path, path) for path in batch["audio"]]
#         return batch
#     dataset = dataset.map(prepend_root, batched=True)

#     test = dataset['test']
#     test = test.rename_column("text_norm", "labels") if use_normalized_label else test.rename_column("text", "labels")
#     test = test.cast_column("audio", Audio(sampling_rate=16_000))
#     # train = train.filter(lambda example: example["duration"] <= 30) #redundant, in clean split.csv they are already filtered

#     _validate_dataset_split(test, "test")
#     return test

def load_train_val_data(dataset_cfg: Any):
    dataset_name = dataset_cfg.get("name")
    if dataset_name not in ["librispeech_clean", "multimed_german", "radmed_uka"]:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from 'librispeech_clean', 'multimed_german' or 'radmed_uka'.")
    
    if dataset_name == "librispeech_clean":
        print("Loading LibriSpeech clean dataset (~120 GB)...")
        return _load_librispeech_dataset_clean()
    elif dataset_name == "multimed_german":
        print("Loading MultiMed German dataset (~5 GB)...")
        return _load_multimed_german()
    elif dataset_name == "radmed_uka":
        root_path = dataset_cfg.get("root_path")
        use_normalizer = dataset_cfg.get("normalize", True)
        apply_replace_dict = dataset_cfg.get("apply_replacements", True)
        datafiles = dataset_cfg.get("datafiles",    {'train':"splits/uka_all_20260121/train.csv", 
                                                    "eval":"splits/uka_all_20260121/valid.csv", 
                                                    "test":"splits/uka_all_20260121/test.csv"})
        
        print("Loading radmed uka dataset from local files...")
        return _load_radmed_uka(root_path=root_path, 
                                datafiles=datafiles, 
                                use_normalized_label=use_normalizer, 
                                apply_replacement_dict=apply_replace_dict,
                                splits=["train", "validation"])
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

def load_test_data(dataset_cfg:Any):    
    dataset_name = dataset_cfg.get("name")
    if dataset_name not in ["librispeech_clean", "multimed_german", "radmed_uka"]:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from 'librispeech_clean', 'multimed_german' or 'radmed_uka'.")
    
    if dataset_name == "radmed_uka":
        root_path = dataset_cfg.get("root_path","/hpcwork/ve001107/uka_dataset/")
        use_normalizer = dataset_cfg.get("normalize", True)
        apply_replace_dict = dataset_cfg.get("apply_replacements", True)
        datafiles = dataset_cfg.get("datafiles",    {'train':"splits/uka_all_20260121/train.csv", 
                                                    "eval":"splits/uka_all_20260121/valid.csv", 
                                                    "test":"splits/uka_all_20260121/test.csv"})
        print("Loading radmed uka dataset from local files...")
        return _load_radmed_uka(root_path=root_path, 
                                datafiles=datafiles,
                                use_normalized_label=use_normalizer, 
                                apply_replacement_dict=apply_replace_dict,
                                splits=["test",])[0]
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")