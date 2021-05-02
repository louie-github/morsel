#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class PCMFormat(object):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        typ=None,
        signed=None,
        bit_depth=None,
        endianness=None,
    ):
        self.name = name
        self.description = description
        self.typ = typ
        self.signed = signed
        self.bit_depth = bit_depth
        self.endianness = endianness

    def __repr__(self) -> str:
        # I'm trying to replicate Python's dataclass __repr__() method,
        # and I'm sure there's a cleaner and faster way to do it, but
        # I can't think of one right now. TODO: Improve this.
        attribs = ["name", "description", "typ", "signed", "bit_depth", "endianness"]
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
PCM_FORMATS = [
    PCMFormat(
        # Unknown format (missing information)
        name="alaw",
        description="PCM A-law",
        typ="alaw",
    ),
    PCMFormat(
        name="f32be",
        description="PCM 32-bit floating-point big-endian",
        typ=float,
        bit_depth=32,
        endianness="big",
    ),
    PCMFormat(
        name="f32le",
        description="PCM 32-bit floating-point little-endian",
        typ=float,
        bit_depth=32,
        endianness="little",
    ),
    PCMFormat(
        name="f64be",
        description="PCM 64-bit floating-point big-endian",
        typ=float,
        bit_depth=64,
        endianness="big",
    ),
    PCMFormat(
        name="f64le",
        description="PCM 64-bit floating-point little-endian",
        typ=float,
        bit_depth=64,
        endianness="little",
    ),
    PCMFormat(
        # Unknown format (missing information)
        name="mulaw",
        description="PCM mu-law",
        typ="mulaw",
    ),
    PCMFormat(
        name="s16be",
        description="PCM signed 16-bit big-endian",
        typ=int,
        signed=True,
        bit_depth=16,
        endianness="big",
    ),
    PCMFormat(
        name="s16le",
        description="PCM signed 16-bit little-endian",
        typ=int,
        signed=True,
        bit_depth=16,
        endianness="little",
    ),
    PCMFormat(
        name="s24be",
        description="PCM signed 24-bit big-endian",
        typ=int,
        signed=True,
        bit_depth=24,
        endianness="big",
    ),
    PCMFormat(
        name="s24le",
        description="PCM signed 24-bit little-endian",
        typ=int,
        signed=True,
        bit_depth=24,
        endianness="little",
    ),
    PCMFormat(
        name="s32be",
        description="PCM signed 32-bit big-endian",
        typ=int,
        signed=True,
        bit_depth=32,
        endianness="big",
    ),
    PCMFormat(
        name="s32le",
        description="PCM signed 32-bit little-endian",
        typ=int,
        signed=True,
        bit_depth=32,
        endianness="little",
    ),
    PCMFormat(
        name="s8",
        description="PCM signed 8-bit",
        typ=int,
        signed=True,
        bit_depth=8,
    ),
    PCMFormat(
        name="u16be",
        description="PCM unsigned 16-bit big-endian",
        typ=int,
        signed=False,
        bit_depth=16,
        endianness="big",
    ),
    PCMFormat(
        name="u16le",
        description="PCM unsigned 16-bit little-endian",
        typ=int,
        signed=False,
        bit_depth=16,
        endianness="little",
    ),
    PCMFormat(
        name="u24be",
        description="PCM unsigned 24-bit big-endian",
        typ=int,
        signed=False,
        bit_depth=24,
        endianness="big",
    ),
    PCMFormat(
        name="u24le",
        description="PCM unsigned 24-bit little-endian",
        typ=int,
        signed=False,
        bit_depth=24,
        endianness="little",
    ),
    PCMFormat(
        name="u32be",
        description="PCM unsigned 32-bit big-endian",
        typ=int,
        signed=False,
        bit_depth=32,
        endianness="big",
    ),
    PCMFormat(
        name="u32le",
        description="PCM unsigned 32-bit little-endian",
        typ=int,
        signed=False,
        bit_depth=32,
        endianness="little",
    ),
    PCMFormat(
        name="u8",
        description="PCM unsigned 8-bit",
        typ=int,
        signed=False,
        bit_depth=8,
    ),
    PCMFormat(
        # Unknown format (missing information)
        name="vidc",
        description="PCM Archimedes VIDC",
        typ="vidc",
    ),
]


STR_TO_PCM_FORMAT = {pcm_format.name: pcm_format for pcm_format in PCM_FORMATS}

# Currently, we support s16le, s24le, and s32le (signed int little-endian)
SUPPORTED_PCM_FORMATS = {
    pcm_format
    for pcm_format in PCM_FORMATS
    if (
        pcm_format.typ == int
        and pcm_format.signed
        and pcm_format.endianness == "little"
    )
}
