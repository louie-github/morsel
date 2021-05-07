#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Currently, this module only supports exporting s16le, s24le, and s32le
# wav files.
DEFAULT_BIT_DEPTH = 16
SUPPORTED_BIT_DEPTHS = {16, 24, 32}


def _int_to_riff(i: int, length: int) -> bytes:
    """Convert an int to its byte representation in a RIFF file.

    Represents integers as unsigned integers in *length* bytes encoded
    in little-endian.

    Args:
        i (int):
            The integer to represent.
        length (int):
            The number of bytes used to represent the integer.

    Returns:
        bytes:
            The bytes representation of the specified integer.
    """
    return i.to_bytes(length, "little", signed=False)


class PCMWaveFileHeader(object):
    """Generates a PCM WAVE file header.

    Attributes:
        num_samples (int):
            The number of samples in the PCM data.
        num_channels (int):
            The number of channels in each sample of the PCM data.
            Defaults to 2.
        sample_rate (int):
            The sample rate of the PCM data measured in cycles per
            second or Hertz (Hz). Defaults to 48000.
        bits_per_sample (int):
            The bit depth or number of bits per sample in the PCM data.
            Defaults to DEFAULT_BIT_DEPTH.
        chunk_id (bytes):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        chunk_size (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        format_ (bytes):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        subchunk_1_id (bytes):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        subchunk_1_size (typ):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        audio_format (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        num_channels (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        sample_rate (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        byte_rate (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        block_align (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        bits_per_sample (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        subchunk_2_id (bytes):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
        subchunk_2_size (int):
            The corresponding RIFF field data (before conversion to
            bytes, if not already)
    """

    def __init__(
        self,
        *,
        num_samples: int,
        num_channels: int = 2,
        sample_rate: int = 48000,
        bits_per_sample: int = DEFAULT_BIT_DEPTH,
    ):
        """Initializes a PCMWaveFileHeader instance.

        Args:
            num_samples (int):
                The number of samples in the PCM data.
            num_channels (int, optional):
                The number of channels in each sample of the PCM data.
                Defaults to 2.
            sample_rate (int, optional):
                The sample rate of the PCM data measured in cycles per
                second or Hertz (Hz). Defaults to 48000.
            bits_per_sample (int, optional):
                The bit depth or number of bits per sample in the PCM data.
                Defaults to DEFAULT_BIT_DEPTH.

        Raises:
            ValueError:
                An unsupported bit depth was specified.
        """
        if bits_per_sample not in SUPPORTED_BIT_DEPTHS:
            raise ValueError(
                " ".join(
                    (
                        "Bit depths other than 16, 24, or 32 are currently",
                        f"not supported (requested {bits_per_sample})",
                    )
                )
            )
        # TODO: Add error-checking (validation) for values such as
        # bounds checking, and multiple-of-8 verification.
        # Calculate some fields which depend on other fields' values
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        subchunk_2_size = num_samples * num_channels * bits_per_sample // 8
        chunk_size = 36 + subchunk_2_size

        # fmt: off
        self.chunk_id        = b"RIFF"
        self.chunk_size      = chunk_size
        self.format_         = b"WAVE"
        self.subchunk_1_id   = b"fmt "
        self.subchunk_1_size = 16
        self.audio_format    = 1  # 1 = PCM
        self.num_channels    = num_channels
        self.sample_rate     = sample_rate
        self.byte_rate       = byte_rate
        self.block_align     = block_align
        self.bits_per_sample = bits_per_sample
        self.subchunk_2_id   = b"data"
        self.subchunk_2_size = subchunk_2_size
        # fmt: on

    def to_bytes(self):
        # Set RIFF fields
        # fmt: off
        header_chunk_id        = self.chunk_id
        header_chunk_size      = _int_to_riff(self.chunk_size, 4)
        header_format_         = self.format_
        header_subchunk_1_id   = self.subchunk_1_id
        header_subchunk_1_size = _int_to_riff(16, 4)
        header_audio_format    = _int_to_riff(1, 2)  # PCM = 1
        header_num_channels    = _int_to_riff(self.num_channels, 2)
        header_sample_rate     = _int_to_riff(self.sample_rate, 4)
        header_byte_rate       = _int_to_riff(self.byte_rate, 4)
        header_block_align     = _int_to_riff(self.block_align, 2)
        header_bits_per_sample = _int_to_riff(self.bits_per_sample, 2)
        header_subchunk_2_id   = self.subchunk_2_id
        header_subchunk_2_size = _int_to_riff(self.subchunk_2_size, 4)
        # fmt: on
        return b"".join(
            (
                header_chunk_id,
                header_chunk_size,
                header_format_,
                header_subchunk_1_id,
                header_subchunk_1_size,
                header_audio_format,
                header_num_channels,
                header_sample_rate,
                header_byte_rate,
                header_block_align,
                header_bits_per_sample,
                header_subchunk_2_id,
                header_subchunk_2_size,
            )
        )
