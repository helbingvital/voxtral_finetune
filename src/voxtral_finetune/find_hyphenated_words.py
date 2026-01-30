import pandas as pd
from pathlib import Path
import re
from voxtral_finetune.normalizer import CustomNormalizer, SpellingNormalizer
import json

path = "splits/uka_all_20260127.csv"

df = pd.read_csv(path, encoding='UTF-8')
print(df.columns)

normalizer = CustomNormalizer(spelling_dict_path=None)
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
                  "doppler", "pkw",
                  "alpha", "beta", "gamma", "sigma",
                  "syndrom", ]
    for word in keep_words:
        if word in part.lower():
            return True

def is_two_medical_fields(word):
    fields = ["behavioral", "dialektisch", "internistisch", "kardiologisch", 
              "plastisch", "chirurgisch", "therapeutisch", "gastroentologisch", "pädiatrisch",
              "klinisch", "neurologisch"
              ]
    parts = word.split('-')
    contained = [False]*len(parts)
    for i, part in enumerate(parts):
        for field in fields:
            if field in part.lower():
                contained[i] = True
    return not (False in contained)

def is_direction(part):
    directions = ["mittig", "außen", "rechts", "links", "unten", "innen", 
                  "flächig", "rundlich", "oval",
                  "kranial", "kaudal", "distal", "proximal",
                  "ventral", "dorsal", "medial", "lateral"]
    for direction in directions:
        if direction in part.lower():
            return True
        
def keep_hyphen_indicator_adjectives(second_part):
    #if second part is one of the following adjectives (+ending) keep hyphen for readability
    indicator_adjectives = ["haltig", "induziert", "basiert", "assoziiert", 
                            "unterstützt", "produzierend", "assistiert", "assoziiert",
                            "bedingt", ]
    for indicator in indicator_adjectives:
        if second_part.lower() in indicator:
            return True
    return False

def handle_adjectives(word):
    '''returns True if the word consists of 2 adjectives that should keep the hyphen'''
    parts = word.split('-')
    if parts[-1][0].isupper():
        #2nd part is substantive
        return False
    #if second part is indicator keep hyphen for readability
    if keep_hyphen_indicator_adjectives(parts[-1]):
        return True    
    #if 2 directions are combined keep hyphen
    elif is_direction(parts[0]) and is_direction(parts[1]):
        return True
    #medizinische Kopplung, z.B. muko-kutan, infero-lateral,... Look for ending with -o
    elif re.search(r'\w+o-', word):
        return True
    return False

def is_english(part):
    english_words = ["whole", "retraining", 
                     "lifestyle", "pigtail", 
                     "tightrope", "triple", 
                     "follow", "child", "seal",
                     "mesh", "bone", "screening",
                     "absence", "score", "plugs", 
                     "ears", "quick", "los", "angeles",
                     "coil", "slow", "waves", "off", "high", 
                     "risk", "sparing", "major", "involved", "cross",
                     "low", "intermediate", "cotton", "wool", "treatment",
                     "air", "aircast", "flow", "upgrade", "upgrades", "years"]
    if part.lower() in english_words:
        return True
    
def is_name(part):
    eigennamen = ["baker", "pylori"]
    if part.lower() in eigennamen:
        return True
    
def should_keep_hyphen(word):
    if word in ["Dialektisch-behaviorale"]:
        print(word)
    # Filtert Zahlen, Sonderzeichen und Quellangaben
    if not re.match(r'^[a-zA-ZäöüÄÖÜß-]+$', word):
        return True
    #word begins or ends with '-'
    pattern = r"^-[\wüöäÜÖÄß]+|[\wüöäÜÖÄß]+-$"
    if re.search(pattern, word):
        return True
    
    if is_two_medical_fields(word):
        return True
    if handle_adjectives(word):
        return True
    #If only one part is capitalized, combination of verb/adjective - noun, ususally needs hyphen
    if is_only_first_part_lower(word):
        return True
    if is_only_second_part_lower(word):
        return True
    parts = word.split('-')
    if keep_hyphen_indicator_adjectives(parts[-1]):
        return True
    for part in parts:
        if is_abbreviation(part):
            return True
        if is_english(part):
            return True
        if keep_hyphen(part):
            return True
        if is_name(part):
            return True
    return False

def create_closed_form(word):
    parts = word.split('-')
    # Erster Teil bleibt, restliche Teile werden kleingeschrieben
    return parts[0] + "".join(p.lower() for p in parts[1:])

# def is_adjective(word):
#     parts = word.split('-')
#     if not parts[-1][0].isupper():
#         return True
#     else:
#         return False
def are_all_parts_lowercase(word):
    parts = word.split('-')
    for part in parts:
        if len(part)==0 or part[0].isupper():
            return False
    return True
    
def is_only_first_part_lower(word):
    parts = word.split('-')
    if parts[1][0].isupper() and parts[0][0].islower():
        return True
    else:
        return False
    
def is_only_second_part_lower(word):
    parts = word.split('-')
    if parts[1][0].islower() and parts[0][0].isupper():
        return True
    else:
        return False

mask = final_list.apply(lambda x: are_all_parts_lowercase(x))
adjectives = final_list[mask]
final_list = final_list[~mask]
mask = adjectives.apply(lambda x: handle_adjectives(x))
adj_def_hyphen = adjectives[mask]
adj_prob_hyphen = adjectives[~mask]
with open("splits/hyphen_rules/adj_with_hyphen.txt", "w") as f:
    f.write("\n".join(adj_def_hyphen))
with open("splits/hyphen_rules/adj_prob_hyphen.txt", "w") as f:
    f.write("\n".join(adj_prob_hyphen))

mask = final_list.apply(lambda x: should_keep_hyphen(x))
no_hyphen = final_list[~mask]
df_dict = pd.DataFrame()
df_dict["original"] = no_hyphen
df_dict["standardized"] = df_dict["original"].apply(lambda x: create_closed_form(x))
# df_dict["adjective"] = df_dict["original"].apply(lambda x: is_adjective(x))
df_dict["first_part_lower"] = df_dict["original"].apply(lambda x: is_only_first_part_lower(x))
# weird_words = df_dict[df_dict["first_part_lower"]==True]["original"].to_list()
# with open("playground/maybe_adj.txt", "w") as f:
#     f.write("\n".join(weird_words))

df_dict.set_index("original", inplace=True)
medical_dict = df_dict["standardized"].to_dict()
sorted_dict = dict(sorted(medical_dict.items()))
with open("splits/hyphen_rules/hyphenated_words_to_normalize.json", 'w',encoding='UTF-8') as f:
    json.dump(sorted_dict, f, indent=2)

keep_hyphen = final_list[mask].sort_values()
keep_hyphen.to_csv("splits/hyphen_rules/hyphenated_words_to_keep.txt",encoding='UTF-8', index=False, header=False)

print("hello")
