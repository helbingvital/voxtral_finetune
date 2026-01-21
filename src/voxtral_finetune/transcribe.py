import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import _load_radmed_uka
from collator import VoxtralCollatorTranscriptionTask

accelerator = Accelerator()
device = accelerator.device
repo_id = "mistralai/Voxtral-Mini-3B-2507"
processor = AutoProcessor.from_pretrained(repo_id)
collator = VoxtralCollatorTranscriptionTask(processor=processor, model_id=repo_id, language="de")

model = VoxtralForConditionalGeneration.from_pretrained(repo_id, device_map=device)

train, eval = _load_radmed_uka(root_path="/hpcwork/ve001107/uka_dataset/")

def collate_fn(examples):
    # Extract the raw audio arrays
    audio_arrays = [x["audio"]["array"] for x in examples]    
    # Apply the transcription request to the whole list at once
    # This handles the padding automatically
    inputs = processor.apply_transcription_request(
        language="en", 
        audio=audio_arrays, 
        model_id=repo_id
    )
    return inputs

#---------Set up Data Collator ----------

dataloader = DataLoader(
    eval, 
    batch_size=4, 
    collate_fn=collator,
    num_workers=2  # Parallel audio decoding
)

model, dataloader = accelerator.prepare(model, dataloader)

# set the language is already know for better accuracy
# inputs = processor.apply_transcription_request(language="de", 
                                                # audio=audio["array"], 
                                                # model_id=repo_id)

# # but you can also let the model detect the language automatically
# inputs = processor.apply_transcription_request(audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", model_id=repo_id) 

# inputs = inputs.to(device, dtype=torch.bfloat16)
# outputs = model.generate(**inputs, max_new_tokens=500)
# decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

# print("\nGenerated responses:")
# print("=" * 80)
# for decoded_output in decoded_outputs:
#     print(decoded_output)
#     print("=" * 80)

model.eval()
print("Starting transcription")
with torch.no_grad():
    for batch in dataloader:
        # Generate
        outputs = model.generate(**batch, max_new_tokens=200)
        
        # Decode only the NEW tokens
        input_len = batch["input_ids"].shape[1]
        decoded = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        
        for text in decoded:
            print(f"Transcription: {text.strip()}")
        break

print("Finished")