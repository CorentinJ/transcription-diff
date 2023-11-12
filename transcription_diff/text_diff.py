import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, overload, Union

import numpy as np
from minineedle import needle

from transcription_diff.text_normalization import normalize_text
from transcription_diff.whisper_asr import whisper_asr
from colorama import Fore as colors


logger = logging.getLogger(__name__)


@dataclass
class TextDiffRegion:
    reference_text: str
    compared_text: str
    pronunciation_match: bool


def clean_text_diff(ref_text: str, compared: str) -> List[TextDiffRegion]:
    alignment = needle.NeedlemanWunsch(ref_text.split(" "), compared.split(" "))
    alignment.align()

    # Arrange
    regions = []
    for ref_word, compared_word in zip(*alignment.get_aligned_sequences()):
        regions.append(TextDiffRegion(
            ref_word if isinstance(ref_word, str) else "",
            compared_word if isinstance(compared_word, str) else "",
            pronunciation_match=(ref_word == compared_word)
        ))

    # Re-add the spaces between words, and prefer to add them on identical regions rather than non-identical ones
    for text_attr in ("reference_text", "compared_text"):
        last_word_region = None
        for region in regions:
            if not getattr(region, text_attr):
                continue
            if last_word_region:
                if last_word_region.pronunciation_match:
                    setattr(last_word_region, text_attr, getattr(last_word_region, text_attr) + " ")
                else:
                    setattr(region, text_attr, " " + getattr(region, text_attr))
            last_word_region = region

    # Compress
    new_regions = []
    for region in regions:
        if new_regions and (new_regions[-1].pronunciation_match == region.pronunciation_match):
            new_regions[-1].reference_text += region.reference_text
            new_regions[-1].compared_text += region.compared_text
        else:
            new_regions.append(region)

    return new_regions


def text_diff(
    reference_texts: Iterable[str], compared_texts: Iterable[str], lang_id: str
) -> List[List[TextDiffRegion]]:
    raw_refs, raw_comps = list(reference_texts), list(compared_texts)

    # Normalize text down to characters that influence pronunciation only
    clean_refs, raw2clean_refs = zip(*[normalize_text(raw_ref, lang_id) for raw_ref in raw_refs])
    clean_comps, raw2clean_comps = zip(*[normalize_text(raw_comp, lang_id) for raw_comp in raw_comps])

    # Align clean texts and isolate errors
    text_diffs = [clean_text_diff(clean_ref, clean_comp) for clean_ref, clean_comp in zip(clean_refs, clean_comps)]

    # Bring the regions up to the unnormalized text space
    for raw_ref, raw2clean_ref, raw_comp, raw2clean_comp, clean_diff in zip(
        raw_refs, raw2clean_refs, raw_comps, raw2clean_comps, text_diffs
    ):
        clean2raw_ref = raw2clean_ref.inverse()
        clean2raw_comp = raw2clean_comp.inverse()

        clean_ref_pos, clean_comp_pos = 0, 0
        raw_ref_pos, raw_comp_pos = 0, 0
        for region in clean_diff:
            # Use slicemaps to figure out which parts of the unnormalized text this region corresponds to
            clean_ref_sli = slice(clean_ref_pos, clean_ref_pos + len(region.reference_text))
            clean_comp_sli = slice(clean_comp_pos, clean_comp_pos + len(region.compared_text))
            if region is not clean_diff[-1]:
                raw_ref_sli = slice(raw_ref_pos, max(clean2raw_ref[clean_ref_sli].stop, raw_ref_pos))
                raw_comp_sli = slice(raw_comp_pos, max(clean2raw_comp[clean_comp_sli].stop, raw_comp_pos))
            else:
                # Ensure we span the entirety of the unnormalized text, slicemaps are not guaranteed to be surjective
                # Typical example: a final punctuation that is erased in text normalization.
                raw_ref_sli = slice(raw_ref_pos, len(raw_ref))
                raw_comp_sli = slice(raw_comp_pos, len(raw_comp))

            # Modify the region in place with the unnormalized text
            region.reference_text = raw_ref[raw_ref_sli]
            region.compared_text = raw_comp[raw_comp_sli]

            # Update the positions
            clean_ref_pos = clean_ref_sli.stop
            clean_comp_pos = clean_comp_sli.stop
            raw_ref_pos = raw_ref_sli.stop
            raw_comp_pos = raw_comp_sli.stop

    return text_diffs


@overload
def transcription_diff(
    text: str, wav: np.ndarray, sr, *, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> List[TextDiffRegion]: ...
@overload
def transcription_diff(
    texts: List[str], wavs: Iterable[np.ndarray], sr, *, audio_lang: str=None, whisper_model_size=2, custom_words=[],
    device="cuda"
) -> List[List[TextDiffRegion]]: ...
@overload
def transcription_diff(
    text: str, fpath: Union[str, Path], *, audio_lang: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> List[TextDiffRegion]: ...
@overload
def transcription_diff(
    texts: List[str], fpaths: Iterable[Union[str, Path]], *, audio_lang: str=None, whisper_model_size=2,
    custom_words=[], device="cuda"
) -> List[List[TextDiffRegion]]: ...
def transcription_diff(
    *args, lang_id: str=None, whisper_model_size=2, custom_words=[], device="cuda"
) -> Union[List[TextDiffRegion], List[List[TextDiffRegion]]]:
    # TODO: doc
    # Arg parsing
    texts, args = args[0], args[1:]
    if single := isinstance(texts, str):
        texts = [texts]

    # Perform ASR
    asr_texts, lang_id = whisper_asr(
        *args, audio_lang=lang_id, whisper_model_size=whisper_model_size, custom_words=custom_words, device=device
    )
    if isinstance(asr_texts, str):
        asr_texts = [asr_texts]

    # Get the diffs
    diffs = text_diff(texts, asr_texts, lang_id)

    if single:
        return diffs[0]
    else:
        return diffs


def render_text_diff(text_diff: List[TextDiffRegion], with_colors=True) -> str:
    str_out = ""
    for region in text_diff:
        if region.pronunciation_match:
            str_out += region.reference_text
        else:
            str_out += "("
            if with_colors:
                str_out += colors.RED
            str_out += region.compared_text
            if with_colors:
                str_out += colors.RESET
            str_out += "|"
            if with_colors:
                str_out += colors.GREEN
            str_out += region.reference_text
            if with_colors:
                str_out += colors.RESET
            str_out += ")"

    return str_out
