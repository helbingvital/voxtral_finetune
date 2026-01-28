import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
from pathlib import Path
from voxtral_finetune.transcribe import compute_metrics_jiwer
from voxtral_finetune.normalizer import CustomNormalizer, SpellingNormalizer

def main():
    p = argparse.ArgumentParser(description="Fine-tune Voxtral")
    p.add_argument("--pred_file", type=str, default=None, help="Path to file containing predictions and label")
    p.add_argument("--split", default='test', type=str, choices=['train', 'valid', 'test', 'all'])
    p.add_argument("--out_dir", type=str, default=None, help="Path to output directory")
    p.add_argument("--normalize", action='store_true', default=False, help="Apply normalizer on predictions.")
    args = p.parse_args()

    pred_file = Path(args.pred_file)
    df = pd.read_csv(pred_file)
    mapping = {"Pred": "pred", "Prediction": "pred", "prediction": "pred", "transcription": "pred", "hyp": "pred",
               "Label": "label", "ref": "label"}
    df = df.rename(columns=mapping)

    out_dir = Path(args.out_dir)
    if args.normalize:
        normalizer = CustomNormalizer()
        df["pred"] = df["pred"].apply(lambda text: normalizer(text))
        spelling_normalizer = SpellingNormalizer()
        df["pred"] = df["pred"].apply(lambda text: spelling_normalizer(text))
        df["label"], df["pred"] = df.apply(lambda row: spelling_normalizer.apply_on_ref_hyp(row["label"], row["pred"]), axis=1)
        out_dir = out_dir / "normalized2"
        df.to_csv(pred_file.parent / f"{args.split}_transcriptions_normalized2.csv")
    else:
        out_dir = out_dir / "not_normalized"
    metrics = compute_metrics_jiwer(df["pred"].to_list(), df["label"].to_list(),out_dir, args.split)

if __name__ == "__main__":
    main()