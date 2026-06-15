from torch.utils.data import Dataset
from torch import from_numpy
from torch import Tensor
import torchaudio 
from pedalboard_utils import * 

import os 
from pathlib import Path
from typing import Tuple, List

# ICMT-SMT-Guitar data offers guitar samples that are monophonic and polyphonic
# it also offers smaples with effects applied, but I'll be working mostly with 
# NoFx samples

IDMT = Path("/home/yuc3/guitar_effects/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS")

class ICMTSMTGuitarDataMono(Dataset): 
    def __init__(self, pedal_dict: dict=None, target_length_seconds:float =2.0):
        self.data_path = IDMT / "Gitarre monophon" 
        self.wav_paths = [path for path in list(self.data_path.glob("*/*/*.wav")) if path.parent.name == "NoFX"]
        self.meta_data = [] 
        self.pedal_dict = pedal_dict
        self.sr = 44100
        self.target_samples = int(target_length_seconds * self.sr)
        
        for wav_path in self.wav_paths:
            codes = wav_path.name.split(".")[0].split("-")
            id_num = codes[-1] 
            
            instrument_type = codes[0][0] 
            instrument_brand = codes[0][1]
            playing_style = codes[0][2] 
            
            midi_number = codes[1][:2]
            string_number = codes[1][2]
            fret_number = codes[1][3:]
            
            self.meta_data.append({
                "path": str(wav_path), 
                "id": id_num,
                "midi": midi_number,
                "string": string_number, 
                "fret": fret_number,
                "instrument": instrument_type,
                "brand": instrument_brand,
                "playing_style": playing_style
            })
        self.n_samples = len(self.wav_paths)
        
    def __getitem__(self, index: int) -> Tuple[Tuple[Tensor, int], Tuple[Tensor,int], List[str]]:
        meta_data = self.meta_data[index]
        waveform, sr = torchaudio.load(meta_data["path"])
        board_string = ""

        wet = None
        # apply effects 
        if self.pedal_dict is not None:
            pedalboard, pedal_string_lists = get_pedalboard_str(self.pedal_dict)
            wet = from_numpy(pedalboard(waveform.numpy(), sr, reset=False))
            wet = wet / wet.abs().max()
            wet = wet[..., 0:self.target_samples]
        
        waveform = waveform / waveform.abs().max()
        waveform = waveform[..., 0:self.target_samples]
            
        return (waveform, sr), (wet, sr), pedal_string_lists
        
    def __len__(self):
        return self.n_samples

class ICMTSMTGuitarDataPoly(Dataset): 
    def __init__(self, pedal_dict=None, target_length_seconds=2.0):
        self.data_path = IDMT / "Gitarre polyphon" 
        self.wav_paths = [path for path in list(self.data_path.glob("*/*/*.wav")) if path.parent.name == "NoFX"]
        self.meta_data = [] 
        self.pedal_dict = pedal_dict
        self.sr = 44100
        self.target_samples = int(target_length_seconds * self.sr)
        
        for wav_path in self.wav_paths:
            codes = wav_path.name.split(".")[0].split("-")
            id_num = codes[-1] 
            
            instrument_type = codes[0][0] 
            instrument_brand = codes[0][1]
            playing_style = codes[0][2] 
        
            midi_number = codes[1][:2] 
            chord_type = codes[1][2:4] 
           
            self.meta_data.append({
                "path": str(wav_path), 
                "id": id_num,
                "midi": midi_number,
                "chord_type": chord_type, 
                "instrument": instrument_type,
                "brand": instrument_brand,
                "playing_style": playing_style
            })
        self.n_samples = len(self.wav_paths)
    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        waveform, sr = torchaudio.load(meta_data["path"])

        wet = None
        # apply effects 
        if self.pedal_dict is not None:
            pedalboard, pedal_string_lists = get_pedalboard_str(self.pedal_dict)
            wet = from_numpy(pedalboard(waveform.numpy(), sr, reset=False))
            wet = wet / wet.abs().max()
            wet = wet[..., 0:self.target_samples]
        
        waveform = waveform / waveform.abs().max()
        waveform = waveform[..., 0:self.target_samples]
            
        return (waveform, sr), (wet, sr), pedal_string_lists
        
    def __len__(self):
        return self.n_samples
