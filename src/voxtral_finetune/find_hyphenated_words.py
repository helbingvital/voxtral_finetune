import pandas as pd
from pathlib import Path
import re
from voxtral_finetune.normalizer import CustomNormalizer, SpellingNormalizer
import json

path = "splits/uka_all_20260127.csv"

df = pd.read_csv(path, encoding='UTF-8')
print(df.columns)

normalizer = CustomNormalizer()
df['text_norm'] = df['text'].apply(lambda x: normalizer(x))
hyphenated_words = df['text_norm'].str.findall(r'\S*-\S*').explode()
hyphenated_words = hyphenated_words.str.strip('.,!?;:"()[]{}\n')
clean_words = hyphenated_words[(hyphenated_words.str.count('-') == 1)& (hyphenated_words.str.contains(r'\d') == False)]
final_list = clean_words.drop_duplicates().dropna()
final_list.to_csv('splits/uka_all_20260127/all_hyphenated_words.txt', index=False, header=False, encoding='UTF-8')
def is_abbreviation(part):
    if re.search(r'[A-Z].*[A-Z]', part):
        return True
    # Mehrere Großbuchstaben hintereinander (z.B. MRT, COVID)
    if re.search(r'[A-Z]{2,}', part):
        return True
    # Einzelbuchstaben (z.B. L-Thyroxin, S-Ca)
    if len(part) == 1 and part.isupper():
        return True
    if len(part) < 3:
        return True
    return False

def keep_hyphen(part):
    keep_words = ["nicht", 
                  "mittig", "außen", "rechts", "links", "unten", "innen", 
                  "flächig", "rundlich", "oval",
                  "doppler", "behaviorale", "dialektisch"]
    if part.lower() in keep_words:
        return True
    
def is_english(part):
    english_words = ["whole", "retraining", 
                     "lifestyle", "pigtail", 
                     "tightrope", "triple", 
                     "follow", "child", "seal",
                     "mesh", "bone", "screening",
                     "absence", "score", "plugs", 
                     "ears", "quick", "los", "angeles",
                     "coil", "slow", "waves", "off", "high", 
                     "risk", "sparing", "major", "involved", "cross"]
    if part.lower() in english_words:
        return True
def should_keep_hyphen(word):
    if word == "-handlungen":
        print(word)
    # Filtert Zahlen, Sonderzeichen und Quellangaben
    if not re.match(r'^[a-zA-ZäöüÄÖÜß-]+$', word):
        return True

    parts = word.split('-')
    for part in parts:
        if is_abbreviation(part):
            return True
        if is_english(part):
            return True
        if keep_hyphen(part):
            return True
    return False

def create_closed_form(word):
    parts = word.split('-')
    # Erster Teil bleibt, restliche Teile werden kleingeschrieben
    return parts[0] + "".join(p.lower() for p in parts[1:])


mask = final_list.apply(lambda x: should_keep_hyphen(x))
no_hyphen = final_list[~mask]

df_dict = pd.DataFrame()
df_dict["original"] = no_hyphen
df_dict["standardized"] = df_dict["original"].apply(lambda x: create_closed_form(x))
df_dict.set_index("original", inplace=True)
medical_dict = df_dict["standardized"].to_dict()
sorted_dict = dict(sorted(medical_dict.items()))
with open("splits/uka_all_20260127/hyphenated_words_to_normalize.json", 'w',encoding='UTF-8') as f:
    json.dump(sorted_dict, f)

keep_hyphen = final_list[mask].sort_values()
keep_hyphen.to_csv("splits/uka_all_20260127/hyphenated_words_to_keep.txt",encoding='UTF-8', index=False, header=False)

print("hello")
