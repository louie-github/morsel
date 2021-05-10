#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Mapping


def _casefold_dict(mapping: Mapping[str, str]):
    return {key.casefold(): value for key, value in mapping.items()}


INTERNATIONAL_MORSE_CODE = _casefold_dict(
    {
        # Letters
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "É": "..-..",  # "accented E"
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        # Figures
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        # Punctuation marks and other signs
        ".": ".-.-.-",
        ",": "--..--",
        ":": "---...",
        "?": "..--..",
        "'": ".----.",
        "-": "-....-",
        "/": "-..-.",
        "(": "-.--.",
        ")": "-.--.-",
        '"': ".-..-.",
        "=": "-...-",
        "+": ".-.-.",
        "×": "-..-",  # X
        "@": ".--.-.",
    }
)

INTERNATIONAL_MORSE_CODE_EXTRA = _casefold_dict(
    {
        "UNDERSTOOD": "...-.",
        "ERROR": "........",
        "INVITATION TO TRANSMIT": "-.-",
        "WAIT": ".-...",
        "END OF WORK": "...-.-",
        "STARTING SIGNAL": "-.-.-",
        "%": "----- -..-. -----",  # 0/0
        "‰": "----- -..-. ----- -----",  # 0/00
    }
)

# Dicts should be casefolded by now.
INTERNATIONAL_MORSE_CODE_FULL = {
    **INTERNATIONAL_MORSE_CODE,
    **INTERNATIONAL_MORSE_CODE_EXTRA,
}
