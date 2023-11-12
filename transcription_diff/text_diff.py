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

    @property
    def is_identical(self) -> bool:
        return self.reference_text == self.compared_text


def clean_text_diff(ref_text: str, compared: str) -> List[TextDiffRegion]:
    alignment = needle.NeedlemanWunsch(ref_text.split(" "), compared.split(" "))
    alignment.align()

    # Arrange
    regions = []
    for ref_word, compared_word in zip(*alignment.get_aligned_sequences()):
        regions.append(TextDiffRegion(
            ref_word if isinstance(ref_word, str) else "",
            compared_word if isinstance(compared_word, str) else "",
        ))
        regions.append(TextDiffRegion(" ", " "))
    regions = regions[:-1]

    # Compress
    new_regions = []
    for region in regions:
        if new_regions and (new_regions[-1].is_identical == region.is_identical):
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
            raw_ref_sli = slice(raw_ref_pos, raw_ref_pos + clean2raw_ref[clean_ref_sli].stop)
            raw_comp_sli = slice(raw_comp_pos, raw_comp_pos + clean2raw_comp[clean_comp_sli].stop)

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
    text: str, wav: np.ndarray, sr, *, audio_lang: str=None, accuracy_mode=2, device="cuda"
) -> List[TextDiffRegion]: ...
@overload
def transcription_diff(
    texts: List[str], wavs: Iterable[np.ndarray], sr, *, audio_lang: str=None, accuracy_mode=2, device="cuda"
) -> List[List[TextDiffRegion]]: ...
@overload
def transcription_diff(
    text: str, fpath: Union[str, Path], *, audio_lang: str=None, accuracy_mode=2, device="cuda"
) -> List[TextDiffRegion]: ...
@overload
def transcription_diff(
    texts: List[str], fpaths: Iterable[Union[str, Path]], *, audio_lang: str=None, accuracy_mode=2, device="cuda"
) -> List[List[TextDiffRegion]]: ...
def transcription_diff(
    *args, lang_id: str=None, accuracy_mode=2, device="cuda"
) -> Union[List[TextDiffRegion], List[List[TextDiffRegion]]]:
    # Arg parsing
    texts, args = args[0], args[1:]
    if single := isinstance(texts, str):
        texts = [texts]

    # Perform ASR
    asr_texts, lang_id = whisper_asr(*args, audio_lang=lang_id, accuracy_mode=accuracy_mode, device=device)
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
        if region.is_identical:
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
