import logging
from functools import lru_cache
from pathlib import Path
from typing import overload, List, Union, Iterable, Tuple

import librosa
import numpy as np
import torch
import whisper
from whisper.audio import SAMPLE_RATE as _WHISPER_SAMPLE_RATE, N_SAMPLES as _WHISPER_CHUNK_SIZE
from whisper.tokenizer import LANGUAGES as _WHISPER_LANGUAGES

from transcription_diff.find_lang_match import find_lang_match


logger = logging.getLogger(__name__)
_WHISPER_LANGUAGES = list(_WHISPER_LANGUAGES)


# TODO: let the user handle the cache
@lru_cache(maxsize=1)
def get_whisper_model(model_size=3, english_only=False, device="cuda"):
    """
    Available models: https://github.com/openai/whisper/blob/main/model-card.md

    :param model_size: controls the accuracy-speed tradeoff. Larger models are slower but more accurate. Ranges from
    1 to 5.
    :param english_only: English-only models can only process input audio and output text in English, but they are
    more accurate. Do not use English-only models for other languages (not even for any-to-English translation) as
    you will get highly inaccurate results.
    """
    model_name = {
        1: "tiny",
        2: "base",
        3: "small",
        4: "medium",
        5: "large",
    }[model_size]
    if english_only and model_size != 5:
        model_name += ".en"

    logger.info(f"Loading whisper model \"{model_name}\" on {device}")
    return whisper.load_model(model_name, device=device)


@overload
def whisper_asr(
    wav: np.ndarray, sr, *, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> Tuple[str, str]: ...
@overload
def whisper_asr(
    wavs: Iterable[np.ndarray], sr, *, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> Tuple[List[str], str]: ...
@overload
def whisper_asr(
    fpath: Union[str, Path], *, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> Tuple[str, str]: ...
@overload
def whisper_asr(
    fpaths: Iterable[Union[str, Path]], *, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> Tuple[List[str], str]: ...
def whisper_asr(
    *args, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> Union[Tuple[str, str], Tuple[List[str], str]]:
    """
    Performs automatic speech recognition on the given audio(s). Supports most languages, and can perform automatic
    language detection.

    :param sr: samples rate of the waveforms, if provided
    :param audio_lang: the lang code of the input audio as an IETF language tag (e.g. "en-us", "fr", ...), if known.
    When None, the language is automatically determined by the model. If provided and the language is English,
    the English-only whisper model will be used.
    :param whisper_model_size: controls the accuracy-speed tradeoff. Ranges from 1 to 5, which 5 being the highest
    accuracy (largest model size) but the lowest inference speed. This parameter has a large impact, consider setting
    it as high as you can afford to.
    :param custom_words: a list of words likely to be unknown to Whisper. We'll attempt to make whisper aware of them
    by passing them to the initial prompt.
    :return: a tuple:
    - The transcription(s) as a string or list of strings
    - The detected language ID of the first sample if <audio_lang> was None, the whisper equivalent of <audio_lang>
    otherwise.
    """
    # Audio args parsing
    if len(args) == 1:
        if single := (isinstance(args[0], str) or isinstance(args[0], Path)):
            fpaths = [args[0]]
        else:
            fpaths = list(args[0])
        # TODO?: batched resampling using torchaudio for efficiency
        wavs = [librosa.core.load(str(fpath), sr=_WHISPER_SAMPLE_RATE)[0] for fpath in fpaths]
        sr = _WHISPER_SAMPLE_RATE
    else:
        wavs, sr = args
        if single := isinstance(wavs, np.ndarray):
            wavs = [wavs]
        wavs = [wav.astype(np.float32) for wav in wavs]

    # Lang args
    if audio_lang:
        lang_idx = find_lang_match(audio_lang, _WHISPER_LANGUAGES)
        if not lang_idx:
            raise ValueError(f"Language code {audio_lang} is not recognized or supported by Whisper.")
        audio_lang = _WHISPER_LANGUAGES[lang_idx[0]]

    # Resample
    # TODO?: batched resampling using torchaudio for efficiency
    wavs = [
        librosa.core.resample(wav, orig_sr=sr, target_sr=_WHISPER_SAMPLE_RATE, res_type="soxr_mq")
        for wav in wavs
    ]

    # Format inputs
    if any(len(wav) > _WHISPER_CHUNK_SIZE for wav in wavs):
        logger.warning(
            # TODO: support for >30s inputs
            "At least one input to whisper is larger than the chunk size (30s), this is not yet supported and the "
            "input will be trimmed."
        )
    wavs = [whisper.pad_or_trim(wav) for wav in wavs]
    mels = torch.stack([whisper.log_mel_spectrogram(wav) for wav in wavs])

    # Ensuring the right device is selected
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available on your torch install, whisper will run on CPU instead. If you do have a "
            "CUDA-compatible GPU available, you may reinstall torch this way to enable CUDA:\n"
            "\tpip uninstall torch\n"
            "\tpip cache purge\n"
            "\tpip install torch -f https://download.pytorch.org/whl/torch_stable.html\n"
        )
        device = "cpu"

    # Inference
    model = get_whisper_model(model_size=whisper_model_size, english_only=(audio_lang == "en"), device=device)
    device = next(model.parameters()).device
    options = whisper.DecodingOptions(
        language=audio_lang,
        # TODO?: support for timestamped ASR
        without_timestamps=True,
        fp16=device.type != "cpu",
        # TODO?: a more reliable way of expecting custom words? Maybe something with beam decoding?
        prompt=f"CUSTOM_WORDS={','.join(custom_words)}" if custom_words else None,
    )
    with torch.inference_mode():
        outputs = model.decode(mels.to(device), options)

    out_lang = audio_lang or outputs[0].language
    if single:
        return outputs[0].text, out_lang
    else:
        return [output.text for output in outputs], out_lang
