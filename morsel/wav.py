#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import io
import itertools

from typing import BinaryIO, Iterable

from .utils import convert_and_warn

# TODO: Add module docstrings, __repr__() methods, and unit tests

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


def generate_pcm_wav_header(
    *,
    num_samples: int,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
):
    """Generates a PCM WAVE (.wav) file header.

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
    num_samples = convert_and_warn(num_samples, int, name="num_samples")
    num_channels = convert_and_warn(num_channels, int, name="num_channels")
    sample_rate = convert_and_warn(sample_rate, int, name="sample_rate")
    bits_per_sample = convert_and_warn(bits_per_sample, int, name="bits_per_sample")

    if bits_per_sample not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            "Bit depths other than 16, 24, or 32 are currently "
            f"not supported (requested {bits_per_sample})"
        )
    # TODO: Add error-checking (validation) for values such as
    # bounds checking, and multiple-of-8 verification.
    # Calculate some fields which depend on other fields' values
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    subchunk_2_size = num_samples * num_channels * bits_per_sample // 8
    chunk_size = 36 + subchunk_2_size

    # Set RIFF fields
    # fmt: off
    header_chunk_id        = b"RIFF"
    header_chunk_size      = _int_to_riff(chunk_size, 4)
    header_format_         = b"WAVE"
    header_subchunk_1_id   = b"fmt "
    header_subchunk_1_size = _int_to_riff(16, 4)
    header_audio_format    = _int_to_riff(1, 2)  # PCM = 1
    header_num_channels    = _int_to_riff(num_channels, 2)
    header_sample_rate     = _int_to_riff(sample_rate, 4)
    header_byte_rate       = _int_to_riff(byte_rate, 4)
    header_block_align     = _int_to_riff(block_align, 2)
    header_bits_per_sample = _int_to_riff(bits_per_sample, 2)
    header_subchunk_2_id   = b"data"
    header_subchunk_2_size = _int_to_riff(subchunk_2_size, 4)
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


class PCMDataGenerator(object):
    def __init__(self, cycle_data: bytes, cycles: int, last_cycle: bytes = b""):
        self.cycle_data = cycle_data
        self.cycles = cycles
        self.last_cycle = last_cycle

    def buffer(self, target_bufsize: int) -> Iterable[bytes]:
        cycles_per_buffer = target_bufsize // len(self.cycle_data)
        if cycles_per_buffer <= 1:
            # Do not bother buffering, return the cycles as they are
            yield from itertools.repeat(self.cycle_data, self.cycles)
            yield self.last_cycle
        else:
            cycles_per_buffer = min(self.cycles, cycles_per_buffer)
            buffer_data = self.cycle_data * cycles_per_buffer
            buffer_cycles, extra_cycles = divmod(self.cycles, cycles_per_buffer)
            extra_cycle_data = self.cycle_data * extra_cycles
            yield from itertools.repeat(buffer_data, buffer_cycles)
            yield extra_cycle_data
            yield self.last_cycle

    def to_bytes(self) -> bytes:
        return b"".join(self.buffer(-1))


# Generators for PCM data (wave functions, silence, etc.)
def generate_silence(
    num_samples: int,
    num_channels: int = 2,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
    **kwargs,
) -> PCMDataGenerator:
    num_samples = convert_and_warn(num_samples, int, name="num_samples")
    num_channels = convert_and_warn(num_channels, int, name="num_channels")
    bits_per_sample = convert_and_warn(bits_per_sample, int, name="bits_per_sample")

    bytes_per_sample = bits_per_sample // 8
    cycle_data = (0).to_bytes(length=bytes_per_sample, byteorder="little", signed=True)
    return PCMDataGenerator(cycle_data=cycle_data, cycles=num_samples * num_channels)


def _clamp_floats(floats: Iterable[float]) -> Iterable[float]:
    for value in floats:
        yield max(min(value, 1.0), -1.0)


def _map_floats_to_ints(floats: Iterable[float], bits_per_sample: int) -> Iterable[int]:
    """Maps float values in the range -1.0 to 1.0 to integer values.

    The integer range is determined by *bits_per_sample*. The range is
    defined as the minimum and maximum values representable using
    signed integers of size *bits_per_sample*.

    Args:
        floats (Iterable[float]):
            The float values to map to integers.
        bits_per_sample (int):
            The bit depth used to determine the output integer range.

    Returns:
        Iterable[int]:
            The mapped integer values.
    """
    max_signed = 2 ** (bits_per_sample - 1) - 1
    min_signed = ~max_signed
    # Every `x` increase above the minimum float value maps to an
    # `x * step_size` increase above the minimum int value
    step_size = (max_signed - min_signed) / (1.0 - (-1.0))
    for value in floats:
        yield round((value - (-1.0)) * step_size + min_signed)


def _repeat_channel_data(data, num_channels: int):
    repeat = itertools.repeat
    for sample in data:
        yield from repeat(sample, num_channels)


def _generate_sine_wave(
    *, angular_frequency: float, amplitude: float, num_samples: int, sample_rate: int
) -> Iterable[float]:
    """Generates samples of a sine wave function.

    Args:
        angular_frequency (float):
            The angular frequency of the function or rate of change in
            radians per second.
        amplitude (float):
            The amplitude of the function or the maximum deviation from
            zero.
        num_samples (int):
            The number of samples to generate.
        sample_rate (int):
            The sample rate of the generated data, in cycles per second
            or Hertz (Hz).

    Returns:
        Iterable[float]:
            The generated samples of the sine wave function.
    """
    sin = math.sin
    for t in range(num_samples):
        yield amplitude * sin(angular_frequency * t / sample_rate)


def generate_sine_wave(
    *,
    frequency: int = 500,
    amplitude: float = 0.75,
    num_samples: int = 48000,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
    allow_clipping=True,
) -> PCMDataGenerator:
    """Generates PCM data for a sine wave. All channels are filled with
    the same data.

    If `allow_clipping = True`, then any floating point values that
    fall outside the allowed range (-1.0 to 1.0) will be clamped to
    either -1.0 or 1.0, whichever is appropriate.

    Each element of the iterator output is one "exact cycle" converted
    into PCM data, except for the last cycle which may be cut off early
    to meet the specified *num_samples*.

    This functions generates samples for its sine wave function by
    generating one "exact cycle" of the function, then repeating the
    samples or data points of that "exact cycle" until the specified
    number of samples is reached.

    An "exact cycle" is the shortest number of cycles of a sine wave
    function whose duration can be represented exactly by the specified
    sample rate.

    For example, a 14Hz sine wave at a 48kHz sample rate needs to cycle
    7 times before it reaches a duration (7/14 = 1/2 second) that can
    be represented exactly by the sample rate (1/2 = 24 000/48 000).

    This avoids unnecessary calculations and extra rounding errors.

    Args:
        frequency (int, optional):
            The frequency of the sine wave in cycles per second or
            Hertz (Hz). Defaults to 500.
        amplitude (float, optional):
            The amplitude of the sine wave or maximum deviation from
            zero. Defaults to 0.75.
        num_samples (int, optional):
            The number of samples to generate. Defaults to 48000.
        num_channels (int, optional):
            The number of channels of the PCM data. Defaults to 2.
        sample_rate (int, optional):
            The sample rate of the PCM data. Defaults to 48000.
        bits_per_sample (int, optional):
            The bit depth / number of bits per sample in the PCM data.
            Defaults to DEFAULT_BIT_DEPTH.
        allow_clipping (bool, optional):
            Whether to allow clipping of output values. Defaults to
            True.

    Raises:
        ValueError:
            An unsupported bit depth was specified.
        ValueError:
            An amplitude smaller than 0.0 was specified.
        ValueError:
            An amplitude larger than 1.0 was specified while clipping
            was not allowed.
        ValueError:
            A frequency was specified that cannot be represented with
            the given sample rate.

    Returns:
        PCMDataGenerator:
            A PCMDataGenerator instance that can generate the final PCM
            data.
    """
    if bits_per_sample not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            "Bit depths other than 16, 24, or 32 are currently not supported "
            + f"(requested {bits_per_sample})"
        )
    if amplitude < 0.0:
        raise ValueError("Given amplitude {amplitude} is smaller than 0.0.")
    elif amplitude > 1.0 and not allow_clipping:
        raise ValueError(
            f"Given amplitude {amplitude} is larger than 1.0 and clipping was "
            + "not allowed."
        )
    if frequency > (sample_rate / 2):
        raise ValueError(
            f"Given frequency {frequency}Hz cannot be represented with sample "
            + f"rate {sample_rate}Hz"
        )
    frequency = convert_and_warn(frequency, int, name="frequency")
    amplitude = convert_and_warn(amplitude, float, name="amplitude")
    num_samples = convert_and_warn(num_samples, int, name="num_samples")
    num_channels = convert_and_warn(num_channels, int, name="num_channels")
    sample_rate = convert_and_warn(sample_rate, int, name="sample_rate")
    bits_per_sample = convert_and_warn(bits_per_sample, int, name="bits_per_sample")
    # "Exact cycle" algorithm
    min_cycles = frequency // math.gcd(frequency, sample_rate)
    # Convert duration of exact cycle to duration in sample rate
    min_samples = sample_rate * min_cycles // frequency
    min_samples = min(min_samples, num_samples)
    angular_frequency = 2 * math.pi * frequency
    exact_cycle = _generate_sine_wave(
        angular_frequency=angular_frequency,
        amplitude=amplitude,
        num_samples=min_samples,
        sample_rate=sample_rate,
    )
    exact_cycle = _clamp_floats(exact_cycle)
    exact_cycle_ints = _map_floats_to_ints(exact_cycle, bits_per_sample=bits_per_sample)
    bytes_per_sample = bits_per_sample // 8
    exact_cycle_bytes = (
        i.to_bytes(bytes_per_sample, "little", signed=True) for i in exact_cycle_ints
    )
    exact_cycle_bytes = _repeat_channel_data(exact_cycle_bytes, num_channels)
    exact_cycle_data = b"".join(exact_cycle_bytes)
    repetitions, last_cycle_length = divmod(num_samples, min_samples)
    last_cycle_data = exact_cycle_data[:last_cycle_length]
    return PCMDataGenerator(exact_cycle_data, repetitions, last_cycle_data)


# Writer functions for PCM data generators
def write_pcm_generator(
    data: PCMDataGenerator, fp: BinaryIO, max_bufsize: int = io.DEFAULT_BUFFER_SIZE
) -> int:
    bytes_written = 0
    for buffer in data.buffer(max_bufsize):
        bytes_written = fp.write(buffer)
    return bytes_written


def write_sine_wave_wav_file(
    fp: BinaryIO,
    buffer_size: int = 8192,
    *,
    frequency: int = 500,
    amplitude: float = 0.75,
    num_samples: int = 48000,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
    allow_clipping=True,
) -> int:
    """Writes the data for a PCM WAV (.wav) file containing a sine wave.

    See `generate_sine_wave()` for more details on generating the sine
    wave data.

    The function will attempt to buffer the individual samples
    generated by `generate_sine_wave()` into chunks with a target size
    of *buffer_size* bytes before writing them into the file, but the
    exact buffering behavior will depend on the bytes objects returned
    by the inner iterator.

    If the inner iterator returns bytes objects that are larger than
    *buffer_size*, *buffer_size* will be ignored and the bytes objects
    will be written to the file directly.

    If the inner iterator returns bytes objects are smaller than
    *buffer_size*, the function will buffer and join as many bytes
    objects as necessary to write a maximum of *buffer_size* bytes to
    the file at once.

    Args:
        fp (BinaryIO):
            The file-like object to write the data to.
        buffer_size (int, optional):
            The target number of bytes to write to the file at a time.
            Defaults to 8192.
        frequency (int, optional):
            The frequency of the sine wave in cycles per second or
            Hertz (Hz). Defaults to 500.
        amplitude (float, optional):
            The amplitude of the sine wave or maximum deviation from
            zero. Defaults to 0.75.
        num_samples (int, optional):
            The number of samples to generate. Defaults to 48000.
        num_channels (int, optional):
            The number of channels of the PCM data. Defaults to 2.
        sample_rate (int, optional):
            The sample rate of the PCM data. Defaults to 48000.
        bits_per_sample (int, optional):
            The bit depth / number of bits per sample in the PCM data.
            Defaults to DEFAULT_BIT_DEPTH.
        allow_clipping (bool, optional):
            Whether to allow clipping of output values. Defaults to True.


    Returns:
        int:
            The number of bytes written.
    """
    header = generate_pcm_wav_header(
        num_samples=num_samples,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
    )
    data = generate_sine_wave(
        frequency=frequency,
        amplitude=amplitude,
        num_samples=num_samples,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        allow_clipping=allow_clipping,
    )
    bytes_written = 0
    bytes_written += fp.write(header)
    bytes_written += write_pcm_generator(data, fp, max_bufsize=buffer_size)
    return bytes_written
