import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml

from dataset import _load_radmed_uka, _convert_radmed_uka_split_csv
from train import train
from main import load_config
print("hello")

# _convert_radmed_uka_split_csv()
# ds_train, ds_validation = _load_radmed_uka()

cfg = load_config("/Users/helbing/Documents/RWTH/Masterarbeit/voxtral/voxtral-finetune/config/voxtral-3B-mini-debug.yaml")
train(cfg, "voxtral-3B-mini-debug")