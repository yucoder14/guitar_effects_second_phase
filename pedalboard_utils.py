import numpy as np 
import random
import math

from functools import partial

from pedalboard import Pedalboard
from pedalboard import Chorus, Distortion, Phaser # guitar-style effects
from pedalboard import Compressor # dynamic range effects
from pedalboard import Reverb # spacial effects 

# probably get rid of this later because I will end up using python3.9+
from typing import Tuple, List

from torch import tensor
from torch import stack

"""
    Helper functions 
"""

def pow_range(lower:int, upper:int, step_size:int) -> np.ndarray: 
    return np.power(
        upper - lower + 1,
        np.arange(step_size) / (step_size - 1)) + lower - 1
    
def lin_range(lower:int, upper:int, step_size:int) -> np.ndarray:  
    return np.linspace(start=lower, stop=upper, num=step_size)

def log_range(lower:int, upper:int, step_size:int) -> np.ndarray: 
    b = (upper - lower) / math.log(step_size)
    return np.log(np.arange(step_size)+1) * b + lower

"""
    Main functions
"""

def create_bins(pedal_dict: dict, step_size: int) -> dict:
    bin_dict = {}
    for effect in pedal_dict.keys(): 
        effect_dict = pedal_dict[effect] 
        params_dict = effect_dict["params"] 
        effect_bin_dict = {}
        params_bin_dict = {}

        effect_bin_dict["pedal"] = effect_dict["pedal"]

        for param in params_dict: 
            param_dict = params_dict[param]
            bins = param_dict["arange"](param_dict["min"], param_dict["max"], step_size)
            params_bin_dict[param] = bins

        effect_bin_dict["params"] = params_bin_dict
        effect_bin_dict["dropout"] = effect_dict["dropout"]

        bin_dict[effect] = effect_bin_dict
    bin_dict["step_size"] = step_size
    return bin_dict


def get_pedalboard(pedal_dict_binned, shuffle: bool=True) -> Pedalboard:
    board = []

    for name, param_dict in pedal_dict_binned.items(): 
        if name == "step_size": 
            continue
        dropout = param_dict['dropout']
        
        if random.random() > dropout:  
            pedal_class = param_dict['pedal']
            params = param_dict['params']
            
            args = {}
            selections = []
            for param, bins in params.items(): 
                i = random.randint(0, step_size - 1)
                args[param] = bins[i] 
            board.append(pedal_class(**args))
    if shuffle: 
        random.shuffle(board)

    board = Pedalboard(board)
        
    return board

def get_pedalboard_str(pedal_dict_binned: dict, shuffle: bool=True) -> Tuple[Pedalboard, List[str]]:
    board = []
    board_string = []
    step_size = pedal_dict_binned["step_size"]

    for name, param_dict in pedal_dict_binned.items(): 
        if name == "step_size": 
            continue
        dropout = param_dict['dropout']
        
        if random.random() > dropout:  
            pedal_class = param_dict['pedal']
            params = param_dict['params']
            
            args = {}
            selections = []
            for param, bins in params.items(): 
                i = random.randint(0, step_size - 1)
                args[param] = bins[i] 
                selections.extend([f"{name}:{param}", str(i)])
            board_string.append(" ".join([name] + selections))
            board.append(pedal_class(**args))
    if shuffle: 
        combined = list(zip(board, board_string))
        if len(combined): 
            random.shuffle(combined)
            board, board_string = zip(*combined)

    board = Pedalboard(board)
        
    return board, list(board_string)

"""
    Constants
"""
    
STEP_SIZE = 10

PEDAL_DICT = {
    "compressor": {
        "pedal": Compressor, 
        "params": {
            "threshold_db": {"min": -60, "max": 10,  "arange": pow_range},
            "ratio":        {"min": 1.1, "max": 20,  "arange": log_range},
            "attack_ms":    {"min": 1,   "max": 100, "arange": log_range},
            "release_ms":   {"min": 40,  "max": 400, "arange": log_range}
        },
        "dropout": 0.5
    },
    "distortion": {
        "pedal": Distortion,
        "params": {
            "drive_db": {"min": 0, "max": 45, "arange": log_range}
        },
        
        "dropout": 0.5
    },
    "chorus": {
        "pedal": Chorus, 
        "params": {
            "rate_hz" :        {"min": 0.1, "max": 8.0,  "arange": log_range}, 
            "centre_delay_ms": {"min": 5,   "max": 25,   "arange": lin_range}, 
            "depth" :          {"min": 0.01, "max": 0.25, "arange": lin_range},
            "feedback" :       {"min": 0.0,  "max": 0.4,  "arange": lin_range},
            "mix" :            {"min": 0.0,  "max": 1.0,  "arange": lin_range} 
        },
        "dropout": 0.5
    },
    "phaser": {
        "pedal": Phaser, 
        "params": {
            "rate_hz":             {"min": 0.1, "max": 5.0,  "arange": log_range}, 
            "depth":               {"min": 0.1, "max": 1.0,  "arange": lin_range}, 
            "centre_frequency_hz": {"min": 100, "max": 2000, "arange": log_range}, 
            "feedback":            {"min": 0.0, "max": 0.6,  "arange": lin_range}, 
            "mix":                 {"min": 0.5, "max": 1.0,  "arange": lin_range}  
        },
        "dropout": 0.5
    },
    "reverb": {
        "pedal": Reverb,
        "params": {
            "room_size": {"min": 0.1,  "max": 1.0, "arange": pow_range}, 
            "damping":   {"min": 0.01, "max": 1.0, "arange": lin_range}, 
            "wet_level": {"min": 0.01, "max": 1.0, "arange": log_range}, 
            "dry_level": {"min": 0.01, "max": 1.0, "arange": log_range}, 
            "width":     {"min": 0.01, "max": 1.0, "arange": lin_range}  
        },
        "dropout": 0.5
    }
}

PEDAL_DICT_BINNED = create_bins(PEDAL_DICT, STEP_SIZE)


"""
    Vocabulary for training
"""

class PedalVocab(object): 
    def __init__(self, padding_idx: int = 0): 
        self.token_to_num = {}
        self.num_to_token = {}
        self.initialized = False
        self.n = 0
        self.padding_idx = padding_idx
        self.token_to_num[str(padding_idx)] = padding_idx

    def __len__(self): 
        return self.n 
        
    def to_num(self, token_str: List[str]) -> List[int]: 
        return list(map(lambda tok: self.token_to_num[tok], token_str))
        
    def to_str(self, token_int: List[int]) -> List[int]: 
        tokens = []
        for num in token_int: 
            if num != self.padding_idx: 
                tokens.append(self.num_to_token[num])
        return tokens

    def initialize(self, pedal_dict: dict, step_size: int) -> None: 
        def _add_token(token):
            self.n += 1
            self.token_to_num[token] = self.n 
            self.num_to_token[self.n] = token 

        for i in range(step_size): 
            _add_token(str(i + 1))
            
        for name, pedal in pedal_dict.items():
            _add_token(name)
            
            for param_name in pedal["params"]:
                _add_token(f"{name}:{param_name}")

        _add_token("<start_order>")
        _add_token("<sep>")
        _add_token("<end>")
        self.initialized = True
        
    def tokenize(self, pedals_str: List[str]) -> List[str]: 
        assert self.initialized, "Must initialize the vocabulary first!"
        
        tokens = ["<start_order>"] 
        params = []
        for pedal_str in pedals_str: 
            pedal_tokens = pedal_str.split(" ")
            tokens.append(pedal_tokens[0])
            params.append("<sep>") 
            params.extend(pedal_tokens[1:]) 
        tokens.extend(params) 
        tokens.append("<end>")

        return tokens
        
VOCAB = PedalVocab()
VOCAB.initialize(PEDAL_DICT, STEP_SIZE)
NUM_VOCAB = len(VOCAB)
MAX_TOKEN_LEN = 5 + 20 * 2 + 2 + 5 # for the case when all effects are present num_types + num_parameters * 2 + num_start_end + num_sep


"""
    Collate function to pad the tokens
"""

def collate_to_len(max_token_len:int, batch: List[Tuple]) -> Tuple: 
    assert isinstance(batch, list)
    dry = [item[0][0] for item in batch] 
    wet = [item[1][0] for item in batch] 

    # this is horrible
    target = [tokens + [0] * (max_token_len - len(tokens)) for item in batch if(tokens:=VOCAB.to_num(VOCAB.tokenize(item[2])))]

    dry_stacked = stack(dry)
    wet_stacked = stack(wet)
    target_stacked = np.stack(target)

    return dry_stacked, wet_stacked, tensor(target_stacked)

collate = partial(collate_to_len, MAX_TOKEN_LEN)