#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict


class PCMFormat(object):
    def __init__(
        self, *, name: str, typ=None, signed=None, bit_depth=None, endianness=None
    ):
        self.name = name
        self.typ = typ
        self.signed = signed
        self.bit_depth = bit_depth
        self.endianness = endianness

    def __repr__(self) -> str:
        # I'm trying to replicate Python's dataclass __repr__() method,
        # and I'm sure there's a cleaner and faster way to do it, but
        # I can't think of one right now. TODO: Improve this.
        attribs = ["name", "typ", "signed", "bit_depth", "endianness"]
        formatted_attribs = []
        for attrib in attribs:
            attrib_value = getattr(self, attrib)
            if attrib_value is not None:
                formatted_attribs.append(f"{attrib}={repr(attrib_value)}")
        return f"PCMFormat({', '.join(formatted_attribs)})"

    def __str__(self) -> str:
        return self.name


# Reference: FFMPEG documentation on Audio Types
# https://trac.ffmpeg.org/wiki/audio%20types
PCM_FORMATS: Dict[str, PCMFormat] = {
    "alaw": PCMFormat(
        typ="alaw",
        # Unknown format (missing information)
        name="PCM A-law",
    ),
    "f32be": PCMFormat(
        typ=float,
        bit_depth=32,
        endianness="big",
        name="PCM 32-bit floating-point big-endian",
    ),
    "f32le": PCMFormat(
        typ=float,
        bit_depth=32,
        endianness="little",
        name="PCM 32-bit floating-point little-endian",
    ),
    "f64be": PCMFormat(
        typ=float,
        bit_depth=64,
        endianness="big",
        name="PCM 64-bit floating-point big-endian",
    ),
    "f64le": PCMFormat(
        typ=float,
        bit_depth=64,
        endianness="little",
        name="PCM 64-bit floating-point little-endian",
    ),
    "mulaw": PCMFormat(
        typ="mulaw",
        # Unknown format (missing information)
        name="PCM mu-law",
    ),
    "s16be": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=16,
        endianness="big",
        name="PCM signed 16-bit big-endian",
    ),
    "s16le": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=16,
        endianness="little",
        name="PCM signed 16-bit little-endian",
    ),
    "s24be": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=24,
        endianness="big",
        name="PCM signed 24-bit big-endian",
    ),
    "s24le": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=24,
        endianness="little",
        name="PCM signed 24-bit little-endian",
    ),
    "s32be": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=32,
        endianness="big",
        name="PCM signed 32-bit big-endian",
    ),
    "s32le": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=32,
        endianness="little",
        name="PCM signed 32-bit little-endian",
    ),
    "s8": PCMFormat(
        typ=int,
        signed=True,
        bit_depth=8,
        name="PCM signed 8-bit",
    ),
    "u16be": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=16,
        endianness="big",
        name="PCM unsigned 16-bit big-endian",
    ),
    "u16le": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=16,
        endianness="little",
        name="PCM unsigned 16-bit little-endian",
    ),
    "u24be": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=24,
        endianness="big",
        name="PCM unsigned 24-bit big-endian",
    ),
    "u24le": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=24,
        endianness="little",
        name="PCM unsigned 24-bit little-endian",
    ),
    "u32be": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=32,
        endianness="big",
        name="PCM unsigned 32-bit big-endian",
    ),
    "u32le": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=32,
        endianness="little",
        name="PCM unsigned 32-bit little-endian",
    ),
    "u8": PCMFormat(
        typ=int,
        signed=False,
        bit_depth=8,
        name="PCM unsigned 8-bit",
    ),
    "vidc": PCMFormat(
        typ="vidc",
        # Unknown format (missing information)
        name="PCM Archimedes VIDC",
    ),
}

_SUPPORTED_PCM_FORMATS = set(["s16le", "s24le", "s32le"])
SUPPORTED_PCM_FORMATS = {
    k: v for k, v in PCM_FORMATS.items() if k in _SUPPORTED_PCM_FORMATS
}
