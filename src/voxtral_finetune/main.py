import sys
import os
import yaml
import argparse
from datetime import datetime

from voxtral_finetune.train import train
from voxtral_finetune.utils import get_abs_project_root_path

DEFAULT_CONFIG_FILE_NAME="config/voxtral-3B-v0.yaml"


def load_config(config_file: str ) -> dict:
    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Config file not found, please try again with a valid path, Exiting...\n{e}")
        sys.exit()
    return cfg


def main():
    p = argparse.ArgumentParser(description="Fine-tune Voxtral")
    p.add_argument("config_name", type=str,  nargs='?', default=None, help="Name of the YAML file in the config folder (without .yaml)")
    args = p.parse_args()

    if args.config_name is None:
        project_dir = get_abs_project_root_path()
        config_file = os.path.join(project_dir, DEFAULT_CONFIG_FILE_NAME)
    else:
        config_file = args.config_name
    
    print("Loading config...")
    cfg = load_config(config_file)
    wandb_name = f"{os.path.basename(config_file).split(".")[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    os.environ["WANDB_PROJECT"] = "Voxtral"
    os.environ["WANDB_NAME"] = wandb_name
    
    print("Starting Train...")
    train(cfg, os.path.basename(config_file).split(".")[0])
    

if __name__ == "__main__":
    main()


#Peft/Lora training https://medium.com/@plawanrath/fine-tuning-mistral-7b-on-apple-silicon-a-mac-users-journey-with-axolotl-lora-c6ff53858e7d
