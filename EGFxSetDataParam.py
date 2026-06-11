from torch.utils.data import Dataset
from torch import from_numpy
import torchaudio 
from pedalboard_param_utils import * 

from torch import tensor

import os 
from pathlib import Path

class EGFxSetData(Dataset): 
    def __init__(self, pedal_dict=None, target_length_seconds=2.0):
        self.data_path = Path("/home/yuc3/guitar_effects/EGFxSetResampled")
        self.wav_paths = list(self.data_path.glob("*/*.wav"))
        self.meta_data = [] 
        self.pedal_dict = pedal_dict
        self.sr = 44100
        self.target_samples = int(target_length_seconds * self.sr)
        
        for wav_path in self.wav_paths: 
            pickup = wav_path.parent.name 

            string, fret = wav_path.name.split(".")[0].split("-")
            self.meta_data.append({
                "path": str(wav_path), 
                "pickup": pickup,
                "string": string, 
                "fret": fret
            })
        self.n_samples = len(self.wav_paths)
    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        waveform, sr = torchaudio.load(meta_data["path"])
        board_string = "<start> clean <end>"

        wet = None
        # apply effects 
        if self.pedal_dict is not None:
            pedalboard, pedal_string_lists = get_pedal_board(self.pedal_dict)
            wet = from_numpy(pedalboard(waveform.numpy(), sr, reset=False))
            board_string = ("<start> " + " <sep> ".join(pedal_string_lists) + " <end>").split(" ")
            wet = wet / wet.abs().max()
            wet = wet[..., 0:self.target_samples]
        
        waveform = waveform / waveform.abs().max()
        waveform = waveform[..., 0:self.target_samples]
            
        return (waveform, sr), (wet, sr), board_string
        
    def __len__(self):
        return self.n_samples