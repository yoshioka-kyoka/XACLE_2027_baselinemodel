"""Stage 1 (docs/strategy.md): extract joint audio+caption features from a Qwen-Omni thinker.

Unlike M2D-CLAP's dual towers (separate encode_audio/encode_text), this puts the
audio and the caption into a single prompt and takes the thinker's own final
hidden state as one joint embedding per pair. Rationale: Ryo Morita's eda004/5
(tech-XACLE-2027) found that comparing *pooled input embeddings* (audio tower
output vs token embedding, averaged, cosine-compared) is ~uncorrelated with
human scores -- the semantic alignment only shows up after the transformer
processes audio and text together, not at the input.

Only the thinker (understanding submodule) is loaded -- never the talker or
code2wav -- so this never touches audio generation, consistent with the
"generation prohibited" condition on every Qwen-Omni entry in the approved
model list.

This is a frozen feature extractor: embeddings are computed once per split and
cached to .npz, then consumed by a lightweight head-training step (see
models/qwen_omni/encoder.py). Re-running the full thinker forward pass inside
every training epoch, as the M2D-CLAP path does, would be wasteful once the
backbone is this large.

Usage:
    python scripts/extract_qwen_omni_embeddings.py \\
        --metadata-csv datasets/.../meta_data/validation_average.csv \\
        --wav-dir datasets/.../wav/validation \\
        --out embeddings/qwen2.5-omni-3b/validation.npz \\
        --max-samples 8   # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

SAMPLE_RATE = 16000
# Qwen2.5-Omni's default system prompt. Supplying it verbatim avoids a
# "System prompt modified" warning from the chat template; irrelevant to the
# result since we never use audio output / generate().
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
PROMPT_TEMPLATE = (
    "Listen to the audio and read the caption. "
    'Caption: "{caption}"\n'
    "Judge how well the audio matches the caption."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--max-sec", type=float, default=10.0, help="Truncate audio, matching baseline max_len")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, help="Limit rows processed (smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def load_audio(wav_path: str, max_sec: float) -> np.ndarray:
    wav, source_sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono
    if source_sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=SAMPLE_RATE)
    max_len = int(max_sec * SAMPLE_RATE)
    return wav[:max_len]


def build_conversation(wav: np.ndarray, caption: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav},
                {"type": "text", "text": PROMPT_TEMPLATE.format(caption=caption)},
            ],
        },
    ]


@torch.no_grad()
def embed_batch(thinker, processor, device, dtype, wavs, captions) -> np.ndarray:
    conversations = [build_conversation(wav, cap) for wav, cap in zip(wavs, captions)]
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False) for conv in conversations
    ]
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(text=texts, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Audio features stay float32 (feature extractor output); everything else matches model dtype.
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype)

    outputs = thinker(**inputs, output_hidden_states=True, use_cache=False)
    last_hidden = outputs.hidden_states[-1]  # (B, T, D)
    mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.float().cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    df = pd.read_csv(args.metadata_csv)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"Embedding {len(df)} pairs from {args.metadata_csv} with {args.model_id} ({args.dtype}, {device})")

    print("Loading processor + thinker (talker/code2wav are never loaded)...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_id, dtype=dtype).to(device)
    thinker.eval()
    hidden_size = thinker.config.text_config.hidden_size
    print(f"Thinker hidden size: {hidden_size}")

    names, embeddings = [], []
    start = time.time()
    for batch_start in range(0, len(df), args.batch_size):
        batch_df = df.iloc[batch_start : batch_start + args.batch_size]
        wavs = [load_audio(str(Path(args.wav_dir) / name.lstrip("/")), args.max_sec) for name in batch_df["wav_file_name"]]
        pooled = embed_batch(thinker, processor, device, dtype, wavs, list(batch_df["text"]))
        names.extend(batch_df["wav_file_name"].tolist())
        embeddings.append(pooled)
        done = batch_start + len(batch_df)
        elapsed = time.time() - start
        print(f"  {done}/{len(df)} ({elapsed:.1f}s, {elapsed / done:.2f}s/sample)")

    embeddings = np.concatenate(embeddings, axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, wav_file_name=np.array(names), embedding=embeddings, model_id=args.model_id)
    print(f"Saved {embeddings.shape} to {out_path}")


if __name__ == "__main__":
    main()
