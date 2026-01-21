import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml

from dataset import _load_radmed_uka, _convert_radmed_uka_split_csv, load_test_data
from train import train
from main import load_config
from collator import VoxtralCollatorTranscriptionTask
from torch.utils.data import DataLoader

from transformers import (
    AutoProcessor,
)
print("hello")

#----------Load Dataset -------------------------
# _convert_radmed_uka_split_csv()
ds_train, ds_validation = _load_radmed_uka(root_path="/hpcwork/ve001107/uka_dataset/")
df_test = load_test_data({'name': "radmed_uka",
  'laguage': "de",
  'root_path': "/Users/helbing/"})
# ---------get sample and batch manually -----------------
first_sample = ds_train[0]
batch = [ds_train[0],ds_train[1],ds_train[2],ds_train[3]]

#---------Set up Data Collator ----------
model_name = "mistralai/Voxtral-Mini-3B-2507"
processor = AutoProcessor.from_pretrained(model_name)
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
