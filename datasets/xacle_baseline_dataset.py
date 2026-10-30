import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import soundfile as sf
import torchaudio

def get_dataset(txt_file_path, wav_dir, max_sec, sr, org_max=10.0, org_min=0.0):
    return XACLEDataset(txt_file_path, wav_dir, max_sec, sr, org_max, org_min)

def get_infdataset(txt_file_path, wav_dir, max_sec, sr):
    return XACLEINFDataset(txt_file_path, wav_dir, max_sec, sr)

def get_cached_embedding_dataset(txt_file_path, embeddings_path, org_max=10.0, org_min=0.0):
    return XACLECachedEmbeddingDataset(txt_file_path, embeddings_path, org_max, org_min)

def get_cached_embedding_infdataset(txt_file_path, embeddings_path):
    return XACLECachedEmbeddingINFDataset(txt_file_path, embeddings_path)

from torch.nn.functional import pad as pad1d 


def load_audio(wav_path):
    """Load a WAV file without relying on TorchCodec/FFmpeg."""
    wav, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T), sample_rate

class XACLEDataset(Dataset):
    def __init__(
            self,
            txt_file_path: str,
            wav_dir : str,
            max_sec: int = 10,
            sr: int = 16_000,
            org_max: float = 10.0,
            org_min: float = 0.0
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.sr = sr
        self.wav_max_len = int(max_sec * sr)
        self.org_mid = (org_max + org_min) / 2
        self.norm_denom = 5.0
        bins = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = [f"{i}-{i+1}" for i in range(0, 10)]
        df["rating_category"] = pd.cut(df["average_score"], bins=bins, labels=labels)
        df["num_class"] = df.groupby("rating_category", observed=False)["rating_category"].transform("count")
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path        = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, source_sr   = load_audio(wav_path)
        wav              = wav.mean(dim=0, keepdim=True)
        if source_sr != self.sr:
            wav = torchaudio.functional.resample(wav, source_sr, self.sr)
        mos             = float(r["average_score"])
        mos_norm        = (mos - self.org_mid) / self.norm_denom
        caption         = r["text"]
        num_class       = int(r["num_class"])

        return dict(
            wav         = wav,
            score       = mos_norm,
            caption     = caption,
            num_class   = num_class,
            wav_path    = wav_path
        )
    
    def collate_fn(self, batch):
        # --- wav padding / trimming --------------------------------------
        wav_max_len = self.wav_max_len
        wavs        = [b["wav"] for b in batch]
        wav_fixed   = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        # --- simple tensors ----------------------------------------------
        mos_score = torch.tensor([b["score"]        for b in batch], dtype=torch.float)
        num_class = torch.tensor([b["num_class"]    for b in batch], dtype=torch.long)

        # M2D-CLAP tokenizes raw captions inside its text encoder.
        captions   = [b["caption"] for b in batch]

        return dict(
            wavs            = wav_batch,
            scores          = mos_score,
            captions        = captions,
            num_class       = num_class,
            wav_paths       = [b["wav_path"] for b in batch]
        )
    
class XACLEINFDataset(Dataset):
    def __init__(
            self,
            txt_file_path: str,
            wav_dir : str,
            max_sec: int = 10,
            sr: int = 16_000
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.sr = sr
        self.wav_max_len = int(max_sec * sr)
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, source_sr = load_audio(wav_path)
        wav = wav.mean(dim=0, keepdim=True)
        if source_sr != self.sr:
            wav = torchaudio.functional.resample(wav, source_sr, self.sr)
        caption  = r["text"]
        return dict(
            wav      = wav,
            caption  = caption,
            wav_path = wav_path
        )
    
    def collate_fn(self, batch):
        # --- wav padding / trimming --------------------------------------
        wav_max_len = self.wav_max_len
        wavs        = [b["wav"] for b in batch]
        wav_fixed   = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        # M2D-CLAP tokenizes raw captions inside its text encoder.
        captions   = [b["caption"] for b in batch]

        return dict(
            wavs            = wav_batch,
            captions        = captions,
            wav_paths       = [b["wav_path"] for b in batch]
        )


def _load_embedding_lookup(embeddings_path):
    """Load a scripts/extract_*_embeddings.py .npz into a {wav_file_name: vector} dict."""
    data = np.load(embeddings_path, allow_pickle=True)
    names = data["wav_file_name"]
    embeddings = data["embedding"]
    return dict(zip(names.tolist(), embeddings))


class XACLECachedEmbeddingDataset(Dataset):
    """Training dataset for a frozen backbone's precomputed joint embeddings.

    Stage 1 (docs/strategy.md): swap only the encoder, keep the normalization,
    class-balance bucketing, loss, and training loop identical to XACLEDataset.
    Used when a backbone (e.g. Qwen-Omni) is too large to re-run every epoch,
    so its embeddings are extracted once (scripts/extract_qwen_omni_embeddings.py)
    and cached instead of recomputed from raw audio on the fly.
    """

    def __init__(self, txt_file_path, embeddings_path, org_max=10.0, org_min=0.0):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.org_mid = (org_max + org_min) / 2
        self.norm_denom = 5.0
        bins = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = [f"{i}-{i+1}" for i in range(0, 10)]
        df["rating_category"] = pd.cut(df["average_score"], bins=bins, labels=labels)
        df["num_class"] = df.groupby("rating_category", observed=False)["rating_category"].transform("count")
        self.df = df
        self.embeddings = _load_embedding_lookup(embeddings_path)
        missing = set(df["wav_file_name"]) - set(self.embeddings)
        if missing:
            raise KeyError(
                f"{len(missing)} wav_file_name(s) in {txt_file_path} have no cached embedding "
                f"in {embeddings_path} (e.g. {sorted(missing)[:3]})"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        mos = float(r["average_score"])
        return dict(
            embedding = self.embeddings[r["wav_file_name"]],
            score     = (mos - self.org_mid) / self.norm_denom,
            num_class = int(r["num_class"]),
            wav_path  = r["wav_file_name"],
        )

    def collate_fn(self, batch):
        embeddings = torch.tensor(np.stack([b["embedding"] for b in batch]), dtype=torch.float)
        mos_score = torch.tensor([b["score"] for b in batch], dtype=torch.float)
        num_class = torch.tensor([b["num_class"] for b in batch], dtype=torch.long)
        return dict(
            embedding = embeddings,
            scores    = mos_score,
            num_class = num_class,
            wav_paths = [b["wav_path"] for b in batch],
        )


class XACLECachedEmbeddingINFDataset(Dataset):
    """Inference counterpart of XACLECachedEmbeddingDataset (no score/num_class)."""

    def __init__(self, txt_file_path, embeddings_path):
        super().__init__()
        self.df = pd.read_csv(txt_file_path)
        self.embeddings = _load_embedding_lookup(embeddings_path)
        missing = set(self.df["wav_file_name"]) - set(self.embeddings)
        if missing:
            raise KeyError(
                f"{len(missing)} wav_file_name(s) in {txt_file_path} have no cached embedding "
                f"in {embeddings_path} (e.g. {sorted(missing)[:3]})"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return dict(embedding=self.embeddings[r["wav_file_name"]], wav_path=r["wav_file_name"])

    def collate_fn(self, batch):
        embeddings = torch.tensor(np.stack([b["embedding"] for b in batch]), dtype=torch.float)
        return dict(
            embedding = embeddings,
            wav_paths = [b["wav_path"] for b in batch],
        )
