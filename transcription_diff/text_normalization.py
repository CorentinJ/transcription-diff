import inspect
import logging
import re
from typing import Tuple, Callable, List

import unicodedata
from langcodes import Language

from transcription_diff.number_normalization import normalize_numbers
from transcription_diff.slice_map import SliceMap


logger = logging.getLogger(__name__)


# Regular expressions matching whitespace. When using with re.split(), the second one will keep whitespaces in the
# output because all captured groups are kept.
_whitespace_excl_re = re.compile(r'\s+')
_whitespace_incl_re = re.compile(r'(\s+)')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile('\\b%s\\.' % abbrev, re.IGNORECASE), expanded)
    for abbrev, expanded in [
        ('mrs', 'misess'),
        ('mr', 'mister'),
        ('dr', 'doctor'),
        ('st', 'saint'),
        ('co', 'company'),
        ('jr', 'junior'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doctors'),
        ('rev', 'reverend'),
        ('lt', 'lieutenant'),
        ('hon', 'honorable'),
        ('sgt', 'sergeant'),
        ('capt', 'captain'),
        ('esq', 'esquire'),
        ('ltd', 'limited'),
        ('col', 'colonel'),
        ('ft', 'feet'),
        ('abbrev', 'abbreviation'),
        ('ave', 'avenue'),
        ('abstr', 'abstract'),
        ('addr', 'address'),
        ('jan', 'january'),
        ('feb', 'february'),
        ('mar', 'march'),
        ('apr', 'april'),
        ('jul', 'july'),
        ('aug', 'august'),
        ('sep', 'september'),
        ('sept', 'september'),
        ('oct', 'october'),
        ('nov', 'november'),
        ('dec', 'december'),
        ('mon', 'monday'),
        ('tue', 'tuesday'),
        ('wed', 'wednesday'),
        ('thur', 'thursday'),
        ('fri', 'friday'),
        ('sec', 'second'),
        ('min', 'minute'),
        ('mo', 'month'),
        ('yr', 'year'),
        ('cal', 'calorie'),
        ('dept', 'department'),
        ('gal', 'gallon'),
        ('kg', 'kilogram'),
        ('km', 'kilometer'),
        ('mt', 'mount'),
        ('oz', 'ounce'),
        ('vol', 'volume'),
        ('vs', 'versus'),
        ('yd', 'yard'),
        ('e\\.g', 'eg'),
        ('i\\.e', 'ie'),
        ('etc', 'etc'),
    ]
]


def expand_abbreviations(text: str):
    orig2new = SliceMap.identity(len(text))
    new_text = text

    for regex, replacement in _abbreviations:
        for match in re.finditer(regex, text):
            new_sli = orig2new[slice(*match.span())]
            new_text = new_text[:new_sli.start] + replacement + new_text[new_sli.stop:]
            orig2new *= SliceMap.identity(new_sli.start) + \
                        SliceMap.lerp(len(match.group()), len(replacement)) + \
                        SliceMap.identity(orig2new.target_len - new_sli.stop)

    return new_text, orig2new


def collapse_whitespace(text: str):
    for part in re.split(_whitespace_incl_re, text):
        match = re.search(_whitespace_excl_re, part)
        if match is not None:
            new_part = re.sub(_whitespace_excl_re, " ", part)
            yield new_part, SliceMap.lerp(len(part), len(new_part))
        else:
            yield part, SliceMap.identity(len(part))


def standardize_characters(text: str):
    for part in re.split(_whitespace_incl_re, text):
        new_part = unicodedata.normalize("NFKC", part)
        transform = SliceMap.lerp(len(part), len(new_part))
        yield new_part, transform


def keep_pronounced_only(text: str):
    kept_idx = [i for i, c in enumerate(text) if c.isalnum() or c in ("-", "'", " ")]
    new_text = "".join(text[i] for i in kept_idx).lower()
    new2orig = SliceMap.from_1to1_map(kept_idx, len(text))
    return new_text, new2orig.inverse()


def apply_text_transforms_with_mapping(
    text: str, funcs: List[Callable], fault_tolerant=False
) -> Tuple[str, SliceMap]:
    """
    :param funcs: a list of Callables that take a text string and return a tuple (new_text, mapping), where the mapping
    must be a SliceMap from the new text to the text provided as argument. For convenience, the function can also be a
    generator function that yields outputs in chunks (new_text_part, mapping_part).
    """
    # Backcompat: we'll support funcs=None as argument
    funcs = funcs or []

    orig2new = SliceMap.identity(len(text))
    for func in funcs:
        # Perform the cleaning operation and obtain the new mapping
        try:
            if inspect.isgeneratorfunction(func):
                new_text = ""
                map_transform = SliceMap.empty()
                for new_text_part, map_transform_part in func(text):
                    new_text += new_text_part
                    map_transform += map_transform_part
            else:
                new_text, map_transform = func(text)
        except Exception as e:
            if fault_tolerant:
                logger.error(f"Exception in cleaning function {func.__name__}: {e}")
                continue
            else:
                raise

        # Update the mapping, verifying that it is valid
        if map_transform.source_len != len(text) or map_transform.target_len != len(new_text):
            if fault_tolerant:
                logger.error("Cleaning operations gave an incorrect mapping")
                map_transform = SliceMap.lerp(len(text), len(new_text))
            else:
                raise RuntimeError("Cleaning operations gave an incorrect mapping")
        orig2new *= map_transform
        text = new_text

    return text, orig2new


def normalize_text(raw_text: str, lang_id: str, fault_tolerant=False) -> Tuple[str, SliceMap]:
    """
    :param fault_tolerant: issues arising in cleaning operations will not raise an exception if True. The cleaning
    and/or mapping may then be incorrect.
    :return: the tuple
    - clean_text: the cleaned text
    - raw2clean: the mapping from raw text to clean text
    """
    # Define the ops to apply
    text_cleaning_ops = [standardize_characters, collapse_whitespace]
    if Language.get(lang_id).language == "en":
        text_cleaning_ops.extend([expand_abbreviations, normalize_numbers])
    text_cleaning_ops.append(keep_pronounced_only)

    return apply_text_transforms_with_mapping(raw_text, text_cleaning_ops, fault_tolerant)
