import os
import pandas as pd
from pathlib import Path
import evaluate
from voxtral_finetune.transcribe import compute_metrics_jiwer
from voxtral_finetune.train import  _build_wer_fn

from transformers import AutoProcessor

# ---------------------- Metrics during Training ---------------------------
#Using evaluate, which usees jiwer under the hood for wer.
#Ensure Versions of evaluate and jiwer are compatible
processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
# metric_file = os.path.join(os.path.dirname(__file__), "wer")
metric_file = "/home/ve001107/voxtral_finetune/voxtral_finetune/src/voxtral_finetune/wer"
metric_fn = _build_wer_fn(processor, evaluate.load(metric_file))


#------------------------ Metrics for transcription -------------------------
#Uses jiwer
prediction_path = Path("/home/ve001107/voxtral_finetune/voxtral_finetune/playground/prediction.csv")
output_dir = Path("playground/test_metrics/")
df_transcriptions = pd.read_csv(prediction_path)

predictions = df_transcriptions.prediction.to_list()
labels = df_transcriptions.label.to_list()

compute_metrics_jiwer(predictions, labels, output_dir, "test")

