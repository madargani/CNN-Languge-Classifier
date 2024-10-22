from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import librosa

class MelSpecDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.files = []
        self.labels = []
        for language in ('ar', 'en', 'zh-TW'):
            files = os.listdir(f'mel_specs/{language}')
            self.files.extend([f'mel_specs/{language}/{file}' for file in files])
            self.labels.extend([language] * len(files))
        self.categories = {'ar': 0, 'en': 1, 'zh-TW': 2}
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        ms = np.load(self.files[idx])
        if self.transforms is not None:
            ms = self.transforms(ms)
        return ms, self.categories[self.labels[idx]]

class AudioDataset(Dataset):
    def __init__(self, sr):
        self.sr = sr
        ar = pd.read_csv('data/ar/validated.tsv', sep='\t', low_memory=False)
        en = pd.read_csv('data/en/validated.tsv', sep='\t', low_memory=False)
        zh = pd.read_csv('data/zh-TW/validated.tsv', sep='\t', low_memory=False)
        self.data = pd.concat([ar, en, zh])[['path', 'locale']]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = f'data/{row['locale']}/clips/{row['path']}'
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        return (y, sr), row['locale']