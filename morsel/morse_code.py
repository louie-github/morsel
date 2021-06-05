#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string

from fractions import Fraction
from typing import (
    BinaryIO,
    Callable,
    Container,
    Dict,
    Iterable,
    Mapping,
    NamedTuple,
    TypeVar,
)

from .mappings import INTERNATIONAL_MORSE_CODE
from .wav import (
    DEFAULT_BIT_DEPTH,
    generate_pcm_wav_header,
    generate_silence,
    generate_sine_wave,
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

X = TypeVar("X")
Y = TypeVar("Y")


# Default settings for Morse code conversion
WORD_SEPARATOR = "/"
LETTER_SEPARATOR = " "
PLACEHOLDER = "?"  # No translation
WHITESPACE = set(string.whitespace)
# Assuming one word = 5 characters
WPM = 15

DEFAULT_DOT_LENGTH = Fraction(1200, WPM)


# Based on the "PARIS" standard; durations in milliseconds (ms)
# Relative lengths and spacing are based off the International Morse
# Code standard (ITU-R M.1677).
class MorseCodeLengths(NamedTuple):
    """All lengths are specified in milliseconds."""

    dot: int
    dash: int
    space_within_letter: int
    space_between_letters: int
    space_between_words: int

    @classmethod
    def from_dot_length(cls, dot_length):
        base_unit = dot_length
        dot = 1 * base_unit
        dash = 3 * base_unit
        space_within_letter = 1 * base_unit
        space_between_letters = 3 * base_unit
        space_between_words = 7 * base_unit
        return cls(
            dot=int(dot),
            dash=int(dash),
            space_within_letter=int(space_within_letter),
            space_between_letters=int(space_between_letters),
            space_between_words=int(space_between_words),
        )


class MorseCodeAudio(NamedTuple):
    dot: bytes
    dash: bytes
    space_within_letter: bytes
    space_between_letters: bytes
    space_between_words: bytes

    @classmethod
    def from_lengths(
        cls,
        lengths: MorseCodeLengths,
        length_factor: int = 1000,
        signal_data_generator: Callable = generate_sine_wave,
        silence_data_generator: Callable = generate_silence,
        frequency: int = 500,
        amplitude: float = 0.75,
        num_channels: int = 2,
        sample_rate: int = 48000,
        bits_per_sample: int = DEFAULT_BIT_DEPTH,
        allow_clipping: bool = True,
    ):
        def _generate_sine_wave_data(num_samples: int):
            return signal_data_generator(
                frequency=frequency,
                amplitude=amplitude,
                num_samples=num_samples,
                num_channels=num_channels,
                sample_rate=sample_rate,
                bits_per_sample=bits_per_sample,
                allow_clipping=allow_clipping,
            ).to_bytes()

        def _generate_silence_data(num_samples: int):
            return silence_data_generator(
                num_samples=num_samples,
                num_channels=num_channels,
                bits_per_sample=bits_per_sample,
            ).to_bytes()

        return cls(
            dot=_generate_sine_wave_data(lengths.dot * sample_rate // length_factor),
            dash=_generate_sine_wave_data(lengths.dash * sample_rate // length_factor),
            space_within_letter=_generate_silence_data(
                lengths.space_within_letter * sample_rate // length_factor
            ),
            space_between_letters=_generate_silence_data(
                lengths.space_between_letters * sample_rate // length_factor
            ),
            space_between_words=_generate_silence_data(
                lengths.space_between_words * sample_rate // length_factor
            ),
        )


def _reverse_dict(dct: Dict[X, Y]) -> Dict[Y, X]:
    return {value: key for key, value in dct.items()}


def _encode(
    text: Iterable[str],
    mapping: Mapping[str, str] = INTERNATIONAL_MORSE_CODE,
    error_on_invalid: bool = False,
    word_separator: str = WORD_SEPARATOR,
    placeholder: str = PLACEHOLDER,
    whitespace: Container[str] = WHITESPACE,
):
    text = "".join(text).casefold()
    word_separator = f" {word_separator} "
    for character in text:
        # Assume that character is valid, then check for whitespace.
        # It should be faster to check for whitespace _after_ trying
        # to translate the character rather than _before_ if the
        # majority of characters are valid and translatable.
        encoded_character = mapping.get(character, None)
        if encoded_character is None:
            if character in whitespace:
                encoded_character = word_separator
            elif error_on_invalid:
                raise KeyError(
                    f"Character {encoded_character} could not be "
                    "found in the given Morse code mapping."
                )
            else:
                encoded_character = placeholder
        yield encoded_character


def encode(
    text: Iterable[str],
    mapping: Mapping[str, str] = INTERNATIONAL_MORSE_CODE,
    join: bool = True,
    error_on_invalid: bool = False,
    letter_separator=LETTER_SEPARATOR,
    word_separator: str = WORD_SEPARATOR,
    placeholder: str = PLACEHOLDER,
    whitespace: Container[str] = WHITESPACE,
):
    output = _encode(
        text=text,
        mapping=mapping,
        placeholder=placeholder,
        error_on_invalid=error_on_invalid,
        word_separator=word_separator,
        whitespace=whitespace,
    )
    output = letter_separator.join(output) if join else output
    return output


def _decode(
    text: str,
    mapping: Mapping[str, str] = _reverse_dict(INTERNATIONAL_MORSE_CODE),
    word_separator: str = WORD_SEPARATOR,
    letter_separator: str = LETTER_SEPARATOR,
    error_on_invalid: bool = False,
    include_empty_letters: bool = False,
    include_empty_words: bool = False,
):
    current_word = []
    for letter in text.split(letter_separator):
        if not letter:
            if include_empty_letters:
                yield letter
            continue
        elif letter == word_separator:
            if current_word or include_empty_words:
                yield "".join(current_word)
                current_word.clear()
            continue
        else:
            decoded_letter = mapping.get(letter, None)
            if decoded_letter is None:
                if error_on_invalid:
                    raise ValueError(
                        f'Could not find a value for Morse Code signal: "{repr(letter)}"'
                    )
            else:
                current_word.append(decoded_letter)
    if current_word or include_empty_words:
        yield "".join(current_word)


# TODO: Use proper function signature rather than *args and **kwargs
def decode(
    text: str,
    mapping: Mapping[str, str] = _reverse_dict(INTERNATIONAL_MORSE_CODE),
    join: bool = True,
    separator: str = " ",
    word_separator: str = WORD_SEPARATOR,
    letter_separator: str = LETTER_SEPARATOR,
    error_on_invalid: bool = False,
    include_empty_letters: bool = False,
    include_empty_words: bool = False,
):
    output = _decode(
        text=text,
        mapping=mapping,
        word_separator=word_separator,
        letter_separator=letter_separator,
        error_on_invalid=error_on_invalid,
        include_empty_letters=include_empty_letters,
        include_empty_words=include_empty_words,
    )
    if join:
        output = separator.join(output)
    return output


def export(
    text: str,
    fp: BinaryIO,
    # generate_sine_wave() keyword arguments
    frequency: int = 500,
    amplitude: float = 0.75,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
    allow_clipping: bool = True,
    # export() function keyword arguments
    word_separator: str = WORD_SEPARATOR,
    letter_separator: str = " ",
    error_on_invalid: bool = True,
    dot_length=DEFAULT_DOT_LENGTH,
) -> int:
    audio_data = MorseCodeAudio.from_lengths(
        MorseCodeLengths.from_dot_length(dot_length),
        length_factor=1000,  # dot_length is in milliseconds
        frequency=frequency,
        amplitude=amplitude,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        allow_clipping=allow_clipping,
    )
    dot_data = audio_data.dot
    dash_data = audio_data.dash
    space_within_letter_data = audio_data.space_within_letter
    space_between_letters_data = audio_data.space_between_letters
    space_between_words_data = audio_data.space_between_words

    # Make room for WAVE header
    fp.seek(44)
    letter_buffer = []
    first_letter = True
    first_word = True
    bytes_written = 0
    for letter in text.split(letter_separator):
        if not letter:
            continue
        if letter == word_separator:
            if not first_word:
                bytes_written += fp.write(space_between_words_data)
            first_word = False
            first_letter = True
        else:
            if not first_letter:
                bytes_written += fp.write(space_between_letters_data)
            first_letter = False
        letter_buffer.clear()
        for character in letter:
            if not character:
                continue
            elif character == ".":
                letter_buffer.append(dot_data)
            elif character == "-":
                letter_buffer.append(dash_data)
            elif error_on_invalid:
                raise ValueError(f"Invalid Morse code character: '{repr(character)}'")
            else:
                pass
        bytes_written += fp.write(space_within_letter_data.join(letter_buffer))
        first_letter = False
    # Write header after writing data (since we don't know how many
    # samples will be written until writing is complete)
    fp.seek(0)
    bytes_written += fp.write(
        generate_pcm_wav_header(
            num_samples=bytes_written // num_channels // (bits_per_sample // 8),
            num_channels=num_channels,
            sample_rate=sample_rate,
            bits_per_sample=bits_per_sample,
        )
    )
    return bytes_written
