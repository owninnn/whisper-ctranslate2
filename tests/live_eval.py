"""
live_wer_eval.py

Evaluate live transcription WER using LibriSpeech (HF).
"""

import numpy as np
from datasets import load_dataset
from jiwer import wer
from typing import List

# Import your Live class
from src.whisper_ctranslate2.live import Live, BlockSize
from src.whisper_ctranslate2.transcribe import TranscriptionOptions


# -----------------------------
# Test wrapper around Live
# -----------------------------
class LiveWER(Live):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_text: List[str] = []

    def process(self):
        if len(self.buffers_to_process) > 0:
            _buffer = self.buffers_to_process.pop(0)

            if not self.transcribe:
                self.transcribe = Transcribe(
                    self.model_path,
                    self.device,
                    self.device_index,
                    self.compute_type,
                    self.threads,
                    self.cache_directory,
                    self.local_files_only,
                    False,
                )

            result = self.transcribe.inference(
                audio=_buffer.flatten().astype("float32"),
                task=self.task,
                language=self.language,
                verbose=False,
                live=True,
                options=self.options,
            )

            text = result["text"].strip()
            if text:
                self.collected_text.append(text)


# -----------------------------
# Streaming simulation
# -----------------------------
def stream_audio(live: LiveWER, audio: np.ndarray, sample_rate: int):
    block_size = int(sample_rate * BlockSize / 1000)

    for i in range(0, len(audio), block_size):
        block = audio[i : i + block_size]
        if block.ndim == 1:
            block = block[:, None]

        live.callback(block, len(block), None, None)
        live.process()

    # Flush remaining audio
    if len(live.buffer) > 0:
        live._save_to_process()
        live.process()


# -----------------------------
# Main evaluation loop
# -----------------------------
def main():
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
    )

    total_wer = []
    max_samples = 50  # adjust for speed

    live = LiveWER(
        model_path="path_to_model",
        cache_directory="cache",
        local_files_only=True,
        task="transcribe",
        language="en",
        threads=4,
        device="cpu",
        device_index=None,
        compute_type="float32",
        verbose=False,
        threshold=0.01,
        input_device=None,
        input_device_sample_rate=16000,
        options=TranscriptionOptions(),
    )

    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference = sample["text"].lower().strip()

        live.collected_text.clear()
        live.buffer = np.zeros((0, 1))
        live.buffers_to_process.clear()
        live.speaking = False
        live.waiting = 0

        stream_audio(live, audio, sr)

        hypothesis = " ".join(live.collected_text).lower().strip()
        sample_wer = wer(reference, hypothesis)

        total_wer.append(sample_wer)

        print(f"[{idx:03d}] WER={sample_wer:.3f}")
        print(f"REF: {reference}")
        print(f"HYP: {hypothesis}")
        print("-" * 60)

    avg_wer = sum(total_wer) / len(total_wer)
    print(f"\nAverage WER over {len(total_wer)} samples: {avg_wer:.3f}")


if __name__ == "__main__":
    main()
