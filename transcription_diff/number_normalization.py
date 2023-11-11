import re

from transcription_diff.slice_map import SliceMap


# This is a set of trivial functions for number normalization. This module does not cover all cases but it's simple
# enough and supports texts mappings.


_comma_number_re = re.compile(r'(\(?[A-Z]{2,3})?([\$|£|¥|€|#|\(]*[0-9][0-9\,\.]+[0-9])([^\s]+)?')
_decimal_number_re = re.compile(r'(number\s)?([0-9]+\.[0-9]+)(\.|,|\?|!)?')
_hash_number_re = re.compile(r'(#)([0-9]+(?:\.[0-9]+)?)(\.|,|\?|!)?')

# currencies
_pounds_re = re.compile(r'(\(?£)([0-9\.]*[0-9]+)(\.|,|\?|\!)?')
_yen_re = re.compile(r'(\(?¥)([0-9]+)(\.|,|\?|\!)?')
_euro_re = re.compile(r'(\(?€)([0-9\.]*[0-9]+)(\.|,|\?|\!)?')
_dollars_re = re.compile(r'(\(?\$)([0-9,]*\.?[0-9]+)([\.|,|\?|\!|\)]+)?')

# currency with abbreviated unit (e.g. B, K, M)
_curr_abbrev_re = re.compile(r'(\(?[$£¥€])([0-9]*\.?[0-9]+)([BKMT]| [BMbmTtr]+illion)([\.|,|\?|\!|\)]+)?')

# units
_ml_re = re.compile(r'([0-9\.]*[0-9]+)(ml)(\.|,|\?|!)?')
_cl_re = re.compile(r'([0-9\.]*[0-9]+)(cl)(\.|,|\?|!)?')
_g_re = re.compile(r'([0-9\.]*[0-9]+)(g)(\.|,|\?|!)?')
_l_re = re.compile(r'([0-9\.]*[0-9]+)(l)(\.|,|\?|!)?')
_m_re = re.compile(r'([0-9\.]*[0-9]+)(m)(\.|,|\?|!)?')
_kg_re = re.compile(r'([0-9\.]*[0-9]+)(kg)(\.|,|\?|!)?')
_mm_re = re.compile(r'([0-9\.]*[0-9]+)(mm)(\.|,|\?|!)?')
_cm_re = re.compile(r'([0-9\.]*[0-9]+)(cm)(\.|,|\?|!)?')
_km_re = re.compile(r'([0-9\.]*[0-9]+)(km)(\.|,|\?|!)?')
_in_re = re.compile(r'([0-9\.]*[0-9]+)(in)(\.|,|\?|!)?')
_ft_re = re.compile(r'([0-9\.]*[0-9]+)(ft)(\.|,|\?|!)?')
_yd_re = re.compile(r'([0-9\.]*[0-9]+)(yd[s]?)(\.|,|\?|!)?')
_s_re = re.compile(r'([0-9\.]*[0-9]+)(s[ecs]*)(\.|,|\?|!)?')

_ordinal_re = re.compile(r'([0-9]+)(st|nd|rd|th)')
_number_re = re.compile(r'([0-9]+)(\.|,|\?|!)?')
_year_re = re.compile(r'([Ff]rom|[Aa]fter|[Bb]efore|[Bb]y|[Uu]ntil)(\s)(?<!\$|£|¥)(1[1-9]|20)([0-9]{2})(\.|,|\?|!)?($|\s)')
_time_re = re.compile(r'([0-2]?[0-9]):([0-9]{2})(am|pm)?(\.|,|\?|!)?($|\s)')

_units = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen'
]

_tens = [
    '',
    'ten',
    'twenty',
    'thirty',
    'forty',
    'fifty',
    'sixty',
    'seventy',
    'eighty',
    'ninety',
]

_digit_groups = [
    '',
    'thousand',
    'million',
    'billion',
    'trillion',
    'quadrillion',
]

_ordinal_suffixes = [
    ('one', 'first'),
    ('two', 'second'),
    ('three', 'third'),
    ('five', 'fifth'),
    ('eight', 'eighth'),
    ('nine', 'ninth'),
    ('twelve', 'twelfth'),
    ('ty', 'tieth'),
]

_sub_ten_nums = ["00", "01",
    "02", "03", "04", "05",
    "06", "07", "08", "09"
]

curr_dict = {
    "$" : "dollars",
    "£" : "pounds",
    "¥" : "yen",
    "€" : "euros"
}

unit_dict = {
    "B" : "billion",
    "K" : "thousand",
    "M" : "million",
    "T" : "trillion"
}


def _remove_commas(text, mapping):
    m = re.findall(_comma_number_re, text)

    # replace each comman with null string
    for result in m:
        r = [''.join(result), ''.join([result[1].replace(",", ""), result[2]])]

        # compute the SliceMap between the numbers
        # with and without numbers
        for i, t in enumerate(mapping):
            if r[0] == t[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    # join normlised text back together
    text_out = ''.join([t[0] for t in mapping])
    return text_out, mapping


def _convert_hash(text, mapping):
    match = re.findall(_hash_number_re, text)

    for result in match:
        # join the found numbers without the hash
        m = ''.join(result)
        r = ' '.join(["number", result[1]])

        # add any following punctuation
        if len(result) == 3:
            r = r + result[2]

        # get the raw and clean parts
        # for computing the mapping
        r = [m, r]
        for i, t in enumerate(mapping):
            if r[0] == t[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))
    text_out = ''.join([t[0] for t in mapping])

    return text_out, mapping


def _expand_decimal_point(text, mapping):
    match = re.findall(_decimal_number_re, text)

    # replace each point in a digit with literal 'point'
    for m in match:
        out = m[0] + m[1].replace('.', ' point ')

        if len(m) == 3:
            out = out + m[2]

        r = [''.join(m), out]

        # compute the SliceMap of the original and normalised digits
        # containing decimal points
        for i, t in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    # join normalised text
    text_out = ''.join([t[0] for t in mapping])
    return text_out, mapping


def _expand_dollars(text, mapping):
    match = re.findall(_dollars_re, text)
    for m in match:
        parts = m[1].split('.')
        if len(parts) > 2:
            out = match + ' dollars'  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            cent_unit = 'cent' if cents == 1 else 'cents'
            out = '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            out = '%s %s' % (dollars, dollar_unit)
        elif cents:
            cent_unit = 'cent' if cents == 1 else 'cents'
            out = '%s %s' % (cents, cent_unit)
        else:
            out = 'zero dollars'

        # append any following punctuation
        if len(m) == 3:
            out = out + m[2]

        # compute the SliceMap between raw and normalised
        r = [''.join(m), out]
        for i, t in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    text_out = ''.join([t[0] for t in mapping])

    return text_out, mapping


def _expand_other_currency(text, mapping, regex, one, many):
    match = re.findall(regex, text)
    for m in match:
        parts = m[1].split(".")
        curr = one if int(parts[0]) == 1 else many
        try:
            out = parts[0] + " " + curr + " " + parts[1]
        except IndexError:
            out = parts[0] + " " + curr
        r = [''.join(m), out]

        # compute the SliceMap
        for i, t in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    text_out = ''.join([t[0] for t in mapping])

    return text_out, mapping


def _expand_abbreviated_currency_unit(text, mapping):
    match = re.findall(_curr_abbrev_re, text)
    # fill this with duplicate words if required
    to_remove = []

    for m in match:
        curr, val, unit, punc = m

        # remove leading paranthesis from currency
        curr = curr.strip("(")

        # split decimal and expand post-decimal digits
        val_parts = val.split(".")
        if len(val_parts) > 1:
            val_out = val_parts[0] + ' ' + 'point ' + ' '.join(val_parts[1])
        else:
            val_out = val_parts[0]

        # reorder elements
        try:
            out = ' '.join([val_out, unit_dict[unit], curr_dict[curr]])
        except KeyError:
            out = ' '.join([val_out, unit, curr_dict[curr]])
        out = out + punc

        ## create raw-clean mapping
        r = [''.join(m), out]

        for i, t in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

            # deal with mapping across  multiple words
            try:
                # check for where the digit is followed by
                # e.g ' billion'
                join_text = ''.join([mapping[i][0], mapping[i+1][0], mapping[i+2][0]])

                if join_text == r[0]:

                    # generate SliceMap of shape
                    # SliceMap(raw_length, clean_length)
                    m = SliceMap.lerp(len(r[0]), len(r[1]))
                    mapping[i] = (out, m)

                    # since upcoming words have been added to
                    # raw-clean mapping, prepare for upcoming
                    # words to be removed from mapping array
                    to_remove.append(i+1)
                    to_remove.append(i+2)
            except IndexError:
                continue

    # remove the duplicate words from mapping array
    new_mapping = [v for i, v in enumerate(mapping) if i not in to_remove]
    text_out = ''.join([t[0] for t in new_mapping])
    return text_out, new_mapping


def _expand_other_unit(text, mapping, regex, one, many):
    match = re.findall(regex, text)
    for m in match:
        # check number for decimal point
        parts = re.split("\.", m[0])
        ## determine if plural unit is needed
        unit = one if parts[0] == "1" else many

        ## check for decimals and read separately
        if len(parts) > 1:
            dec = ''.join([i + " " for i in parts[1]])

            ## convert decimal into point inside function
            ## to maintain the mapping
            parts = parts[0] + " " + "point" + " " + dec

            ## always use plural unit if theres a decimal
            unit = many
            out = parts + unit
        else:
            out = parts[0] + " " + unit

        out = out + m[-1]
        ## create raw-clean mapping
        r = [''.join(m), out]

        for i, t in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    text_out = ''.join([t[0] for t in mapping])
    return text_out, mapping


def _expand_year(text, mapping):
    # FIXME: "In 1950 cases..." will be incorrectly expanded as a year
    m = re.findall(_year_re, text)
    for result in m:
        prep, mill_cent, dec_year, post = str(result[0]), str(result[2]), str(result[3]), str(result[4])
        if mill_cent == "20" and dec_year in _sub_ten_nums:
            year_out = mill_cent + dec_year
        elif dec_year in _sub_ten_nums:
            year = dec_year[-1]
            year_out = mill_cent + " " + "oh" + " " + year
        else:
            year_out = _number_to_words(int(mill_cent)) + " " + _number_to_words(int(dec_year))
        year_out = year_out + post

        # compute the SliceMap between raw and normalised years
        r = [mill_cent + dec_year + post, year_out]
        for i, t in enumerate(mapping):
            if t[0] == r[0] and mapping[i-2][0] == prep:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    text_out = ''.join([t[0] for t in mapping])

    return text_out, mapping


def _expand_time(text, mapping):
    m = re.findall(_time_re, text)
    for result in m:
        hour, minute = result[0], result[1]

        # remove the leading zero for hours like "09"
        hour = hour.strip("0")

        # remove zeros when there are no following minutes
        # and convert leading zeros to "oh"
        if minute == "00":
            minute = ""
        elif minute[0] == "0":
            minute = ' '.join(["oh", minute[1]])

        # check for a following am/pm
        if result[2] != "":
            am_pm = ' '.join(result[2])

            out = " ".join([hour, minute, am_pm])
        else:
            out = " ".join([hour, minute])

        # add following puncuation
        if result[3] != "":
            out = out + result[3]

        # compute SliceMap
        r = [''.join([result[0], ":", result[1], result[2], result[3]]), out]
        for i, t, in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    text_out = ''.join(t[0] for t in mapping)
    return text_out, mapping


def _standard_number_to_words(n, digit_group):
    parts = []
    if n >= 1000:
        # Format next higher digit group.
        parts.append(_standard_number_to_words(n // 1000, digit_group + 1))
        n = n % 1000

    if n >= 100:
        parts.append('%s hundred' % _units[n // 100])
    if n % 100 >= len(_units):
        parts.append(_tens[(n % 100) // 10])
        parts.append(_units[(n % 100) % 10])
    else:
        parts.append(_units[n % 100])
    if n > 0:
        parts.append(_digit_groups[digit_group])
    return ' '.join([x for x in parts if x])


def _number_to_words(n):
    # Handle special cases first, then go to the standard case:
    if n >= 1000000000000000000:
        return str(n)  # Too large, just return the digits
    elif n == 0:
        return 'zero'
    elif n % 100 == 0 and n % 1000 != 0 and n < 3000:
        return _standard_number_to_words(n // 100, 0) + ' hundred'
    else:
        return _standard_number_to_words(n, 0)


def _expand_number(text, mapping):
    match = re.findall(_number_re, text)
    for m in match:

        out = _number_to_words(int(m[0])) + m[1]
        r = [''.join(m), out]

        for i, t in enumerate(mapping):

            # only compare the numeric portion of the string
            text_re_nums = re.search(r"\d+", t[0])
            if text_re_nums is None:
                continue

            if r[0] == text_re_nums.group():

                # escape regex special characters (e.g., question mark)
                rep = re.escape(r[0])
                j = re.sub(rep, r[1], t[0])
                mapping[i] = (j, SliceMap.lerp(t[1].source_len, len(j)))

    text_out = ''.join([t[0] for t in mapping])
    return text_out, mapping


def _expand_ordinal(text, mapping):
    match = re.findall(_ordinal_re, text)
    for m in match:
        num = _number_to_words(int(m[0]))
        for suffix, replacement in _ordinal_suffixes:
            if num.endswith(suffix):
                out = num[:-len(suffix)] + replacement
                break
            else:
                out = num + 'th'
        r = [''.join(m), out]

        # compute SliceMap
        for i, t in enumerate(mapping):
            if t[0] == r[0]:
                mapping[i] = (r[1], SliceMap.lerp(t[1].source_len, len(r[1])))

    text_out = ''.join([t[0] for t in mapping])

    return text_out, mapping


def normalize_numbers(text: str):
    # N.B.: all number norm functions use an old word-level mapping convention, hence we have convert our inputs
    # and outputs to that format
    words = re.split("(\s+)", text)
    mapping = list(zip(words, [SliceMap.identity(len(word)) for word in words]))

    text, mapping = _remove_commas(text, mapping)
    text, mapping = _expand_year(text, mapping)
    text, mapping = _expand_abbreviated_currency_unit(text, mapping)
    text, mapping = _expand_other_currency(text, mapping, _pounds_re, "pound", "pounds")
    text, mapping = _expand_other_currency(text, mapping, _yen_re, "yen", "yen")
    text, mapping = _expand_other_currency(text, mapping, _euro_re, "euro", "euros")
    text, mapping = _expand_other_unit(text, mapping, _ml_re, "milliliter", "milliliters")
    text, mapping = _expand_other_unit(text, mapping, _cl_re, "centiliter", "centiliters")
    text, mapping = _expand_other_unit(text, mapping, _g_re, "gram", "grams")
    text, mapping = _expand_other_unit(text, mapping, _kg_re, "kilogram", "kilograms")
    text, mapping = _expand_other_unit(text, mapping, _mm_re, "millimeter", "millimeters")
    text, mapping = _expand_other_unit(text, mapping, _cm_re, "centimeter", "centimeters")
    text, mapping = _expand_other_unit(text, mapping, _km_re, "kilometer", "kilometers")
    text, mapping = _expand_other_unit(text, mapping, _in_re, "inch", "inches")
    text, mapping = _expand_other_unit(text, mapping, _ft_re, "foot", "feet")
    text, mapping = _expand_other_unit(text, mapping, _l_re, "liter", "liters")
    text, mapping = _expand_other_unit(text, mapping, _m_re, "meter", "meters")
    text, mapping = _expand_other_unit(text, mapping, _yd_re, "yard", "yards")
    text, mapping = _expand_other_unit(text, mapping, _s_re, "second", "seconds")
    text, mapping = _expand_dollars(text, mapping)
    text, mapping = _convert_hash(text, mapping)
    text, mapping = _expand_decimal_point(text, mapping)
    text, mapping = _expand_time(text, mapping)
    text, mapping = _expand_ordinal(text, mapping)
    text, mapping = _expand_number(text, mapping)

    raw2clean_map = SliceMap.empty()
    for word, word_map in mapping:
        raw2clean_map += word_map
    return text, raw2clean_map
