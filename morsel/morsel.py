#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string

from fractions import Fraction
from typing import Iterable, List, Mapping, Union

from .mappings import (
    INTERNATIONAL_MORSE_CODE,
    INTERNATIONAL_MORSE_CODE_FULL,
)

"""
Reference: ITU-R M.1677: International morse code
https://www.itu.int/dms_pubrec/itu-r/rec/m/R-REC-M.1677-1-200910-I!!PDF-E.pdf

This script can either translate strings via a "naive" one-to-one
correspondence / mapping to International Morse Code signals, or by
following some of the standard's additional guidelines for the
transmission of certain sequences, e.g. transmitting the sequence 4%
as 4-0/0 and 5½‰ as 5-1/2-0/00, among other things.

However, this script *does not* claim compliance with the
aforementioned ITU-R M.1677 standard for International Morse Code, and
should not be construed or understood in any way to be as such.
"""

# Default settings for Morse code conversion
WORD_SEPARATOR = "/"
PLACEHOLDER = "?"  # No translation
WHITESPACE = set(string.whitespace)
# Assuming one word = 5 characters
WPM = 15

# Based on the "PARIS" standard; durations in milliseconds (ms)
# Relative lengths and spacing are based off the International Morse
# Code standard (ITU-R M.1677).
DOT_LENGTH = Fraction(1200, WPM)
DASH_LENGTH = 3 * DOT_LENGTH
WITHIN_LETTER_SPACE_LENGTH = 1 * DOT_LENGTH
LETTER_SPACE_LENGTH = 1 * DOT_LENGTH
WORD_SPACE_LENGTH = 1 * DOT_LENGTH


def _encode(
    text: Iterable[str],
    mapping: Mapping[str, str] = INTERNATIONAL_MORSE_CODE,
    placeholder: str = PLACEHOLDER,
    error_on_invalid=False,
    word_separator: str = WORD_SEPARATOR,
    whitespace=WHITESPACE,
):
    text = "".join(text).casefold()
    for character in text:
        if character in whitespace:
            yield word_separator
            continue
        encoded_character = mapping.get(character, None)
        if encoded_character is None:
            if error_on_invalid:
                raise KeyError(
                    f"Character {encoded_character} could not be "
                    "found in the given Morse code mapping."
                )
            else:
                encoded_character = placeholder
        yield encoded_character


def encode(*args, join=True, **kwargs):
    output = _encode(*args, **kwargs)
    output = " ".join(output) if join else output
    return output
