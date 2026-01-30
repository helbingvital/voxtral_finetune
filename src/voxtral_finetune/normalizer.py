import re
import unicodedata
from collections.abc import Iterator
from fractions import Fraction
from re import Match
from typing import Optional, Union, List
from enum import Enum
import jiwer
import json

class NormalizerMode(Enum):
    '''OFF: don't perform normalization, load original label,
     LOAD: Use the normalized labels from split.csv,
     NORMALIZE: (Re)Apply normalizer when loading a sample, necessary if Normalizer was updated since creation of split.csv'''
    OFF = 0
    LOAD = 1 
    NORMALIZE = 2

class CustomNormalizer:
    def __init__(self, spelling_dict_path: Optional[str]="splits/hyphen_rules/hyphenated_words_to_normalize.json"):
        self.normalizers = [ NumberNormalizer(), 
                            UnitNormalizer(), 
                            SpellingNormalizer(path_substitute_dict=spelling_dict_path)]
    def __call__(self, text:str):
        for normalizer in self.normalizers:
            text = normalizer(text)
        return text
    def get_info(self):
        info = ""
        for normalizer in self.normalizers:
            info += normalizer.get_name()
            info += ", "
    
    
class UnitNormalizer:
    def __init__(self):
        self.basic_units = {
            "liter": "l", 
            "meter": "m",
            "gramm": "g",
            "sekunde": "s",
            "mol": "mol",
            "unit": "U",
            "units": "U"
        }
        self.units_without_prefix = {
            "minute": "min",
            "minuten": "min",
            "stunde": "h",
            "stunden": "h",
            "tag": "Tag", 
            "ie": "IE"
        }
        self.small_prefixes = {
            "dezi": "d",
            "milli": "m",
            "mikro": "µ",
            "nano": "n", 
        }

        self.volume_units = {
            "kubikcentimeter": "cm³",
            "kubikzentimeter": "cm³",
            "kubikmeter": "m³"
        }
        

        #concatenate to obtain all units in combination with all possible prefixes
        self.units_with_prefix = {}
        for prefix_key, prefix_value in self.small_prefixes.items():
            for unit_key, unit_value in self.basic_units.items():
                new_key = prefix_key+unit_key
                new_value = prefix_value + unit_value
                self.units_with_prefix[new_key] = new_value
            #for special case of U, transcription model might insert spaces...
            self.units_with_prefix[prefix_key + " unit"] = prefix_value+'U'
            self.units_with_prefix[prefix_key + " units"] = prefix_value+'U'
            self.units_with_prefix[prefix_key + " U"] = prefix_value+'U'

        self.units_with_prefix["kilometer"] = "km"
        self.units_with_prefix["kilogramm"] = "kg"

        self.all_units = self.basic_units | self.units_with_prefix | self.units_without_prefix | self.volume_units
        self.all_units_abbrev_pattern =  "(" + "|".join(re.escape(u) for u in self.all_units.values()) + ")" 
      
    @staticmethod
    def get_name():
        return "UnitNormalizer"

    def __call__(self, text:str):
        '''
        Replace unit words with their abbreviations:
        - always use abbreviation for unitwords with prefix (milli, kilo, ...) (milligramm -> mg)
        - replace unitword with abbreviation if following a digit (1 Liter -> 1 l)
        - add space between digit and unit abbreviation (1mg -> 1 mg)
        - replace all unitwords with abbreviation if "/ unitword" or "pro unitword" (1 gramm pro deziliter -> 1 gramm/dl)
        - replace "d" with "Tag" if after "pro" or "/"
        - replace "pro abbreviation" with "/abbrev.
        - remove space if / is following unit abbreviation or digit
        - replace dL with dl
        - add space between digit and %
        - replace "1 bis 2" with "1-2", "1 zu 2" with "1/2"
        '''
        #TODO: probably there shouldn't be a whitespace but \u202f which is narrow no-break space
        #always use abbreviation for units with prefix
        for key, abbreviation in {**self.units_with_prefix, **self.volume_units}.items(): 
            pattern = re.compile(rf"\b{key}(?=\b|$)", re.IGNORECASE)
            match = pattern.search(text)
            # if match:
            #     print(match)
            text = re.sub(pattern, abbreviation, text)

        #replace all units if following a digit
        for key, abbreviation in (self.all_units).items():
            pattern = re.compile(rf"\b(\d+)\s*{key}(?=\b|$)", re.IGNORECASE)
            replacement = rf"\1 {abbreviation}"
            text = re.sub(pattern, replacement, text)

        #add space for all unitabbreviations following a digit
        pattern = re.compile(rf"\b(\d+)\s*({self.all_units_abbrev_pattern})(?=\b|$)")
        text = pattern.sub(r"\g<1> \g<2>", text)

        #replace all unitwords with abbreviation if "/ unitword" or "pro unitword"
        for key, abbreviation in (self.all_units).items():
            pattern = re.compile(rf"(\bpro|/)\s*{key}(?=\b|$)", re.IGNORECASE)
            replacement = rf"/{abbreviation}"
            text = re.sub(pattern, replacement, text)

        #replace "d" with "Tag" if after "pro" or "/"
        text = self._replace_d_with_Tag(text)

        #replace "pro abbreviation" with "/abbrev."
        pattern = re.compile(rf"\b(pro)\s*({self.all_units_abbrev_pattern})(?=\b|$)")
        text = pattern.sub(r"/\g<2>", text)

        #remove space if unit abbreviation is following /
        pattern = re.compile(rf"\b(/)\s*{self.all_units_abbrev_pattern}(?=\b|$)")
        text = pattern.sub(r"\g<1>\g<2>", text)

        #remove space if / is following unit abbreviation
        pattern = re.compile(rf"\b{self.all_units_abbrev_pattern}\s*(/)(?=\b|$)")
        text = pattern.sub(r"\g<1>\g<2>", text)

        #remove space if / is following digit
        pattern = re.compile(rf"\b(\d+)\s*(/)")
        text = pattern.sub(r"\g<1>/", text)

        #replace dL with dl
        pattern = re.compile(rf"\b(dL)(?=\b|$)")
        text = pattern.sub(r"dl", text)
        
        text = self._normalize_percent(text)
        text = self._replace_bis_zu(text)
        return(text)


    def _replace_d_with_Tag(self, text:str):
        pattern = re.compile(rf"(\bpro|/)\s*(d)(?=\b|$)", re.IGNORECASE)
        text = pattern.sub(r"/Tag", text)
        return text
        # /d -> /Tag
        # pro d -> /Tag

    def _remove_space_before_unit(self, text:str):
        pattern = re.compile(rf"\b(\d+)\s+{self.all_units_abbrev_pattern}(?=\b|$)")
        text = pattern.sub(r"\g<1>\g<2>", text)
        return text
    
    def _normalize_percent(self, text:str):
        pattern = re.compile(rf"\b(\d+)\s*(Prozent|%)", re.IGNORECASE)
        text = pattern.sub(r"\g<1> %", text)
        return text
    
    def _replace_bis_zu(self, text:str):
        #X bis X -> X-X
        pattern = re.compile(rf"\b(\d+)\s*(bis)\s*(\d+)", re.IGNORECASE)
        text = pattern.sub(r"\g<1>-\g<3>", text)
        #X zu X -> X/X
        pattern = re.compile(rf"\b(\d+)\s*(zu)\s*(\d+)", re.IGNORECASE)
        text = pattern.sub(r"\g<1>/\g<3>", text)
        return text

    def print(self, path):
        pass
    #print all possible replacements -> json dump

    def pretty_print(self):
        pass#print all replacements structured in their groups
   
class NumberNormalizer:
    '''
    zweijähriges -> 2-jähriges
    zwei-jähriges -> 2-jähriges   
    '''
    @staticmethod
    def get_name():
        return "NumberNormalizer"

    def __init__(self):
        self.numbers = {
            name: i for i, name in enumerate(
                [
                    "eins",
                    "zwei",
                    "drei",
                    "vier",
                    "fünf",
                    "sechs",
                    "sieben",
                    "acht",
                    "neun",
                    "zehn",
                    "alf",
                    "zwölf",
                    "dreizehn",
                    "vierzehn",
                    "fünfzehn",
                    "sechzehn",
                    "siebzehn",
                    "achtzehn",
                    "neunzehn",
                ],
                start=1,
            )        
        }
        self.numbers_pattern =  "(" + "|".join(re.escape(u) for u in self.numbers.keys()) + ")" 
        self.roman = {'1':'I', '2':'II', '3':'III', '4':'IV', '5':'V', '6':'VI', '7':'VII', '8':'VIII'}

    def __call__(self, text:str):
        #replace zweijähriges -> 2-jähriges
        for key, value in self.numbers.items():
            pattern = re.compile(rf"\b({key})(-jährig|jährig)", re.IGNORECASE)
            text = pattern.sub(rf"{value}-jährig", text)

        #replace 2 jähriges -> 2-jähriges
        pattern = re.compile(rf"\b(\d+)\s*(jährig)", re.IGNORECASE)
        text = pattern.sub(r"\g<1>-jährig", text)

        #TODO: Lebersegment 5 -> Lebersegment V
        #           Segment 5 -> Segment V
        #Soll das ueberhaupt?
        return text
    
class SpellingNormalizer:
    @staticmethod
    def get_name():
        return "SpellingNormalizer"
    def __init__(self, 
                 path_substitute_dict: Optional[str]="splits/uka_all_20260127/hyphenated_words_to_normalize.json"):
        #wrong_spelling: normalized spelling
        # self.graphemes = {"zi": "ti",
        #                   "f": "ph"}

        # Attention, case sensitive!
        self.words = {"ggf.": "gegebenenfalls",
                      "A.": "Ateria",
                      "M.": "Musculus",
                      "Kalzium": "Calcium",
                      "thoracalis": "thorakalis",
                      }
        self.word_parts = {"ograph": "ograf", #e.g. Sonografie, Mammografie,
                           "graphie": "grafie",
                           "graphisch": "grafisch",
                           "zial": "tial"}
        # self.no_hyphen_words = {"Thorax-Röntgen": "Thoraxröntgen",
        #                         "Mamma-Sonographie": "Mammasonografie", #can also just be part of longer word, eg. Mammasonografien
        #                         "Methylprednisolon-Gabe": "Methylprednisolongabe"
        #                         }

        if path_substitute_dict:
            with open(path_substitute_dict) as f:
                self.substitute_dict = json.load(f)
                self.words = {**self.words, **self.substitute_dict}
        self.word_pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in self.words.keys()) + r')\b')
    
    def __call__(self, text:str):
        # for original, replacement in self.words.items():
        #     pattern = re.compile(rf"\b{original}\b", re.IGNORECASE) #only if match is a complete word
        #     text = pattern.sub(rf"{replacement}", text)
        for original, replacement in self.word_parts.items():
            pattern = re.compile(rf"{original}")
            text = pattern.sub(rf"{replacement}", text)
        text = self.word_pattern.sub(lambda x: self.words[x.group(0)], text)
        
        return text
        

    # def apply_on_ref_hyp(self, ref: str, hyp:str, alignments: Optional[List[jiwer.AlignmentChunk]] = None):
    #     #Bindestriche?!
    #     if not alignments:
    #         out_words = jiwer.process_words(ref, hyp)
    #         alignments = out_words.alignments[0]
    #     new_ref = ref
    #     new_hyp = hyp
    #     for alignment in alignments:
    #         if alignment.type=="substitute" and alignment.ref_end_idx-alignment.ref_start_idx==1 and alignment.hyp_end_idx-alignment.hyp_start_idx==1:
    #             ref_word = ref.split()[alignment.ref_start_idx]
    #             hyp_word = hyp.split()[alignment.hyp_start_idx]
    #             for original, standard in self.graphemes.items():
    #                 # if original in hyp_word and original not in ref_word:
    #                 #     hyp_word = hyp_word.replace(original, standard)
    #                 # if original in ref_word and original not in hyp_word:
    #                 #     ref_word = ref_word.replace(original, standard)
    #                 if original in hyp_word or original in ref_word:
    #                     hyp_word = hyp_word.replace(original, standard)
    #                     ref_word = ref_word.replace(original, standard)
    #                     if hyp_word == ref_word:
    #                         new_ref.replace(ref.split()[alignment.ref_start_idx], ref_word)
    #                         new_hyp.replace(hyp.split()[alignment.hyp_start_idx], hyp_word)
    #     if len((new_ref, new_hyp))!=2:
    #         print(new_hyp)
    #         print(new_ref)
    #     return new_ref, new_hyp
        
                    


        

# spelling = SpellingNormalizer()
# refs = ["Duplexsonographie zeigte eine hochgradige Stenose der A. femoralis superficialis.",
#        "Angiographie der unteren Extremitäten"]
# hyps = ["Duplexsonographie zeigte eine hochgradige Stenose der Arteria femoralis superficialis.",
#        "Angiografie der unteren Extremitäten"]

# new_refs, new_hyps = [],[]
# for ref, hyp in zip(refs, hyps):
#     new_refs.append(spelling(ref))
#     new_hyps.append(spelling(hyp))
#     spelling.apply_on_ref_hyp(ref, hyp)