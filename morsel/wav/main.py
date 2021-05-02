#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# TODO: Add docstrings


from typing import Iterable, Union

from .pcm_formats import PCMFormat, SUPPORTED_PCM_FORMATS


def _int_to_field(i: int, length: int):
    return i.to_bytes(length, "little", signed=False)


class PCMWaveFileHeader(object):
    # fmt: off
    _EMPTY = b"\x00"
    chunk_id        = b"RIFF"
    chunk_size      = _EMPTY * 4  # 36 + SubChunk2Size (or file size - 8 bytes)
    format_         = b"WAVE"  # format_ to avoid conflict with builtin "format"
    subchunk_1_id   = b"fmt "
    subchunk_1_size = _int_to_field(16, 4)
    audio_format    = _int_to_field(1, 2)  # PCM = 1
    num_channels    = _EMPTY * 2
    sample_rate     = _EMPTY * 4
    byte_rate       = _EMPTY * 4  # SampleRate * NumChannels * BitsPerSample / 8
    block_align     = _EMPTY * 2  # NumChannels * BitsPerSample / 8 (one sample length)
    bits_per_sample = _EMPTY * 2  # Bit depth (e.g. 16 bit, 24 bit)
    subchunk_2_id   = b"data"
    subchunk_2_size = _EMPTY * 4  # NumSamples * NumChannels * BitsPerSample / 8
    # Alias Python variables to RIFF field names
    ChunkID       = chunk_id
    ChunkSize     = chunk_size
    Format        = format_
    Subchunk1ID   = subchunk_1_id
    Subchunk1Size = subchunk_1_size
    AudioFormat   = audio_format
    NumChannels   = num_channels
    SampleRate    = sample_rate
    ByteRate      = byte_rate
    BlockAlign    = block_align
    BitsPerSample = bits_per_sample
    Subchunk2ID   = subchunk_2_id
    Subchunk2Size = subchunk_2_size
    # fmt: on

    def __init__(
        self,
        *,
        num_channels: int,
        sample_rate: int,
        bits_per_sample: int,
        num_samples: int,
    ):
        # TODO: Add error-checking (validation) for values such as
        # bounds checking, and multiple-of-8 verification.
        # Calculate some fields which depend on other fields' values
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        subchunk_2_size = num_samples * num_channels * bits_per_sample // 8
        chunk_size = 36 + subchunk_2_size
        # Set non-constant fields
        self.chunk_size = _int_to_field(chunk_size, 4)
        self.num_channels = _int_to_field(num_channels, 2)
        self.sample_rate = _int_to_field(sample_rate, 4)
        self.byte_rate = _int_to_field(byte_rate, 4)
        self.block_align = _int_to_field(block_align, 2)
        self.bits_per_sample = _int_to_field(bits_per_sample, 2)
        self.subchunk_2_size = _int_to_field(subchunk_2_size, 4)

    def to_bytes(self) -> bytes:
        return b"".join(
            (
                self.chunk_id,
                self.chunk_size,
                self.format_,
                self.subchunk_1_id,
                self.subchunk_1_size,
                self.audio_format,
                self.num_channels,
                self.sample_rate,
                self.byte_rate,
                self.block_align,
                self.bits_per_sample,
                self.subchunk_2_id,
                self.subchunk_2_size,
            )
        )


class PCMWaveFileData(object):
    def __init__(
        self,
        data: Union[Iterable[int], Iterable[float]],
        pcm_format: Union[str, PCMFormat] = "s24le",
    ):
        if not isinstance(pcm_format, PCMFormat):
            pcm_format_obj = SUPPORTED_PCM_FORMATS.get(pcm_format, None)
            if pcm_format_obj is None:
                raise NotImplementedError(
                    " ".join(
                        f"PCM format {pcm_format} is either an invalid format",
                        "or is not yet currently supported.",
                    )
                )
            else:
                pcm_format = pcm_format_obj
