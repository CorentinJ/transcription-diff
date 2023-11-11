from typing import Union, List

from langcodes import Language


def find_lang_match(
    req_lang: Union[str, Language], avail_langs: Union[List[str], List[Language]], territory_match=True,
) -> List[int]:
    """
    Find the best match for a requested language in a list of available languages.

    This method uses the langcode library to deal with the many ways of specifiying languages and the many variations
    they can have. See https://pypi.org/project/langcodes/ for more information.

    :param req_lang: the language requested, as a language code or Language instance
    :param avail_langs: the list of available languages, as language codes or Language instances
    :param territory_match: whether to also match the territory (~= accent) of the language.
        - If <req_lang> has a territory specified and this argument is True, only entries that specifically match the
          requested territory will be considered.
        - In any other case, the territory will be ignored and only the language will be considered.
    :return: a list of indices of the qualifying matches in <avail_langs>, possibly empty. All matches are considered
    equally good.
    """
    if isinstance(req_lang, str):
        req_lang = Language.get(req_lang)
    if isinstance(avail_langs[0], str):
        avail_langs = [Language.get(lang) for lang in avail_langs]

    # Filter languages that don't match the requested language
    match_idx = [i for i, avail_lang in enumerate(avail_langs) if avail_lang.language == req_lang.language]

    # Also filter by territory if applicable
    if territory_match and req_lang.territory is not None:
        match_idx = [i for i in match_idx if avail_langs[i].territory == req_lang.territory]

    return match_idx
