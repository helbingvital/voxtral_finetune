import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml

from dataset import _load_radmed_uka, _convert_radmed_uka_split_csv, load_test_data, load_train_val_data
from train import train
from main import load_config
from collator import VoxtralCollatorTranscriptionTask
from torch.utils.data import DataLoader

from transformers import (
    AutoProcessor,
    VoxtralProcessor
)
print("hello")

#----------Load Dataset -------------------------
# _convert_radmed_uka_split_csv(split_csv="./splits/uka_all_20260121.csv", new_split_csv_directory="./splits/uka_all_20260121")
cfg = load_config("/home/ve001107/voxtral_finetune/voxtral_finetune/config/voxtral-3B-mini-debug.yaml")
ds_train, ds_validation = load_train_val_data(cfg["dataset"])
ds_test = load_test_data(cfg["dataset"])
print(len(ds_train), len(ds_validation), len(ds_test))

# ---------get sample and batch manually -----------------
first_sample = ds_train[0]
batch = [ds_train[0],ds_train[1],ds_train[2],ds_train[3]]

#---------- test apply_transcription_request (inside Collator)-------------
model_name = "mistralai/Voxtral-Mini-3B-2507"
processor = VoxtralProcessor.from_pretrained(model_name)
audio_arrays = [item["audio"]["array"] for item in batch]
tokenized_transcription_request = processor.apply_transcription_request(
            # language=self.language,
            audio=first_sample["audio"]["array"],
            sampling_rate=16000,
            format=["wav"] * 1,
            model_id=model_name, 
            language="de"
        )

#---------Set up Data Collator ----------
collator = VoxtralCollatorTranscriptionTask(processor=processor, model_id=model_name, language="de")
first_sample_collate = collator([first_sample,])
batch_feature = collator(batch)

#------- Use DataCollator with Dataloader -------------------
dataloader = DataLoader(
    ds_validation, 
    batch_size=4, 
    collate_fn=collator,
    num_workers=2  # Parallel audio decoding
)

for batch in dataloader:
    print(len(batch))
    break


#----------Set up Trainer (also loads dataset as specified in config)--------------
# cfg_path="/home/ve001107/voxtral_finetune/voxtral_finetune/config/voxtral-3B-mini-debug.yaml"
# cfg = load_config(cfg_path)
# train(cfg, "voxtral-3B-mini-debug")
