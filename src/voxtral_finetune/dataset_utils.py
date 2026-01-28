import pandas as pd
import jiwer
from pathlib import Path
from voxtral_finetune.normalizer import CustomNormalizer

def get_uid_from_label(label, df, key="label") -> str:
    mask = df[key].str.contains(label, case=False, na=False)
    found_samples = df.loc[mask]
    if len(found_samples) == 1:
        return found_samples.iloc[0].name
    else: #either 0 or more than 1 hit
        return ""

# -------------------------- Functions operating on transcriptions/alignments ----------------------------------------------------
def suspect_hallucination(transcription, label, insertions_threshold=2):
    if len(transcription.split()) >= len(label.split())+insertions_threshold:
        return True
    else: 
        return False
    
def deletion_at_beginning(alignments):
    first_chunk = alignments[0]
    if first_chunk.type == 'delete':
        num_deletions = first_chunk.ref_end_idx - first_chunk.ref_start_idx
    else:
        num_deletions = 0
    return num_deletions

def deletion_at_end(alignments):
    last_chunk = alignments[-1]
    if last_chunk.type == 'delete':
        num_deletions = last_chunk.ref_end_idx - last_chunk.ref_start_idx
    else:
        num_deletions = 0
    return num_deletions

def detect_cutoff_audio(out_dir, 
                        data, 
                        labels: list, 
                        preds: list,
                        threshold: int=1, 
                        run_name='',                                                                                       
                        out_words: jiwer.WordOutput|None = None,
                        save_alignments: bool = False,
                        alignment_dir = None,
                        key_df_uids = "text_norm"):
    '''
    Take a baseline prediction and use it to find samples with possibly missing audio at end (recording stopped too early) or beginning (recording started too late).  

    Inputs:  
        out_dir:                    str or Path to save "samples_deletion_at_end_threshX.csv".  
        data:                       Dataframe containing columns 'label' and 'uid', needed to find the uid corresponding to transcription.  
        labels, preds:              used to find samples that miss transcribed words in end or beginning.  
        threshold:                  int, the threshold of deletions at end/beginning of a transcription to flag sample.  
        out_words:                  if provided skip recalculation of jiwer.process_words.  
        save_alignments:            bool, if True creates "{run_name}_alignment_threshX.txt" and "{run_name}_error_freq_threshX.txt" from detected samples        
        alignment_dir:              Directory to save alignments, uses result_dir if not provided. 

    Returns:  
    Number of samples with suspected missing beginning/end'''
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = CustomNormalizer()
    labels = [normalizer(x) for x in labels]
    preds = [normalizer(x) for x in preds]
    if not out_words:
        out_words = jiwer.process_words(reference=labels, hypothesis=preds)

    hit_idx_end, hit_idx_start = [],[]
    for index, alignment in enumerate(out_words.alignments):
        num_deletions_end = deletion_at_end(alignment)
        if num_deletions_end >= threshold:
            hit_idx_end.append(index)
        num_deletions_begin = deletion_at_beginning(alignment)
        if num_deletions_begin>=threshold:
            hit_idx_start.append(index)

    labels_hit_end = [labels[i] for i in hit_idx_end]
    preds_hit_end = [preds[i] for i in hit_idx_end]
    labels_hit_start = [labels[i] for i in hit_idx_start]
    preds_hit_start = [preds[i] for i in hit_idx_start]

    if save_alignments:  
        alignment_dir = Path(alignment_dir) if alignment_dir else out_dir
        alignment_dir.mkdir(parents=True, exist_ok=True)        

        
        out_words_hits_end = jiwer.process_words(labels_hit_end, preds_hit_end)
        hits_alignment_end = jiwer.visualize_alignment(out_words_hits_end)
        hits_error_freq_end = jiwer.visualize_error_counts(out_words_hits_end)
        
        out_words_hits_start = jiwer.process_words(labels_hit_start, preds_hit_start)
        hits_alignment_start = jiwer.visualize_alignment(out_words_hits_start)
        hits_error_freq_start = jiwer.visualize_error_counts(out_words_hits_start)

        with open(alignment_dir / f"Cutoff_audio_{run_name}_alignment.txt", 'w') as f:
            f.write(f"Detected {len(hit_idx_start)}samples with missing Audios at beginning.\n")
            f.write(hits_alignment_start)
            f.write(f"Detected {len(hit_idx_end)}samples with missing Audios at end.\n")
            f.write(hits_alignment_end)
        with open(alignment_dir / f"Cutoff_audio_{run_name}_error_freq.txt", 'w') as f:
            f.write(f"Detected {len(hit_idx_start)}samples with missing Audios at beginning.\n")
            f.write(hits_error_freq_start)
            f.write(f"Detected {len(hit_idx_end)}samples with missing Audios at end.\n")
            f.write(hits_error_freq_end)

    #get uids from critical samples
    # hits_org_labels = [label_org[i] for i in hit_idx_end]
    # hits_org_pred = [pred_org[i] for i in hit_idx_end]
    rows = []
    counter_not_found = 0
    for label, pred in zip(labels_hit_end, preds_hit_end):
         try: #Some samples might been transcribed in baseline, but have been removed from split for other reasons.
            uid = get_uid_from_label(label, data, key_df_uids)
            row = {"uid": uid, "do_not_use":True, "reason":"Probably Audio is cut of in end", "label":label, "base-prediction": pred}
            rows.append(row)
         except:
             counter_not_found += 1
             pass
    print(f"{counter_not_found} samples couldnt be found.")
    for label, pred in zip(labels_hit_start, preds_hit_start):
        try: #Some samples might been transcribed in baseline, but have been removed from split for other reasons.
            uid = get_uid_from_label(label, data, key_df_uids)
            row = {"uid": uid, "do_not_use":True, "reason":"Probably Audio is cut of in beginning", "label":label, "base-prediction": pred}
            rows.append(row)
        except:
            counter_not_found += 1
            pass
         
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"samples_cutoff_audio_thresh{threshold}.csv")

    return len(hit_idx_end)+len(hit_idx_start)

 

def find_suspected_hallu(run_name: str, 
                         hallu_thresh: int, 
                         results_dir: Path, 
                         all_predictions: pd.DataFrame,
                         data: pd.DataFrame,
                         save_alignments: bool = False):
    '''Do a simple check to see whether predicted num words is larger by threshold than label num words.  
    Save word alignment and error_freq of critical samples in results_dir.  
    Returns number of suspected hallucinations.  
    Inputs:  
        all_predictions: pd.Dataframe() with columns 'Pred', 'Label'
        data: pd.DataFrame() describing the dataset, with columns 'label' and 'uid'
    '''
    results_dir.mkdir(parents=True, exist_ok=True)
    all_predictions["suspect_hallu"] = all_predictions.apply(lambda row: suspect_hallucination(row['Pred'], row['Label'], hallu_thresh), axis=1)
    print(f"Hallucinations suspected in {all_predictions.suspect_hallu.sum()} samples")
    df_suspect_hallu = all_predictions[all_predictions['suspect_hallu']]
    df_suspect_hallu["uid"] = df_suspect_hallu["Label"].apply(lambda label: get_uid_from_label(label, data, "text_norm"))
    df_suspect_hallu.to_csv(results_dir / f"detected_hallus_thresh_{hallu_thresh}_{run_name}.csv")
    
    if save_alignments:
        out_words = jiwer.process_words(reference=df_suspect_hallu['Label'].tolist(), 
                                    hypothesis=df_suspect_hallu['Pred'].tolist())
        alignment = jiwer.visualize_alignment(out_words)
        error_freq = jiwer.visualize_error_counts(out_words)
        with open(results_dir / f"hallus_thresh_{str(hallu_thresh)}_{run_name}_alignment.txt", 'w') as f:
            f.write(alignment)
        with open(results_dir / f"hallus_thresh_{str(hallu_thresh)}_{run_name}_error_freq.txt", 'w') as f:
            f.write(error_freq)
    
    return len(df_suspect_hallu)


df_pred = pd.read_csv("results/voxtral-3B-mini-baseline_0123/all_transcriptions.csv")
df_dataset = pd.concat([pd.read_csv("splits/uka_all_20260127/train.csv"), pd.read_csv("splits/uka_all_20260127/valid.csv"), pd.read_csv("splits/uka_all_20260127/test.csv")])
df_dataset.set_index("uid", inplace=True)
detect_cutoff_audio("splits/uka_all_20260121", 
                    df_dataset, 
                    df_pred["Label"].to_list(),
                    df_pred["Pred"].to_list(),
                    run_name="voxtral_mini_baseline_0122",
                    save_alignments=True,
                    alignment_dir="results/voxtral-3B-mini-baseline_0123",
                    key_df_uids="text_norm")

find_suspected_hallu(run_name="voxtral_mini_baseline_0122",
                     hallu_thresh=2,
                     results_dir=Path("splits/uka_all_20260121"),
                     all_predictions=df_pred,
                     data = df_dataset,
                     save_alignments=True)