import librosa
import numpy as np
import pytest

from transcription_diff.whisper_asr import whisper_asr


@pytest.mark.parametrize("audio_lang", ["en-gb", "fr-CA", None])
@pytest.mark.parametrize("custom_words", [[], ["butterfly"]])
@pytest.mark.parametrize("whisper_model_params", [
    dict(whisper_model_size=1, device="cpu"),
    dict(whisper_model_size=2, device="cuda"),
])
def test_whisper_asr_args(audio_lang, custom_words, whisper_model_params):
    # Single in-memory input
    sample_rate = 32000
    wav = np.random.randn(sample_rate * 4)
    asr_out, audio_lang_out = whisper_asr(
        wav, sample_rate, audio_lang=audio_lang, **whisper_model_params, custom_words=custom_words
    )
    assert audio_lang is None or audio_lang_out == audio_lang[:2]

    # Multiple in-memory input
    sample_rate = 22500
    wavs = [np.random.randn(sample_rate * 4) for _ in range(3)]
    asr_out, audio_lang_out = whisper_asr(
        wavs, sample_rate, audio_lang=audio_lang, **whisper_model_params, custom_words=custom_words
    )
    assert audio_lang is None or audio_lang_out == audio_lang[:2]

    # One file on disk
    asr_out, audio_lang_out = whisper_asr(
        librosa.example("libri1"), audio_lang=audio_lang, **whisper_model_params, custom_words=custom_words
    )
    assert audio_lang is None or audio_lang_out == audio_lang[:2]

    # Multiple files on disk
    asr_out, audio_lang_out = whisper_asr(
        [librosa.example("libri1"), librosa.example("libri2"), librosa.example("libri3")],
        audio_lang=audio_lang, **whisper_model_params, custom_words=custom_words
    )
    assert audio_lang is None or audio_lang_out == audio_lang[:2]
