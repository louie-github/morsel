#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import gcd, pi, sin
from itertools import chain, repeat
from typing import BinaryIO, Iterable, Iterator

from .wavefile import (
    DEFAULT_BIT_DEPTH,
    SUPPORTED_BIT_DEPTHS,
    PCMWaveFileHeader,
)

# TODO: Add module docstrings, __repr__() methods, and unit tests


class PCMDataGenerator(object):
    def __init__(self, cycle_data: bytes, cycles: int, last_cycle: bytes = b"") -> None:
        self.cycle_data = cycle_data
        self.cycles = cycles
        self.last_cycle = last_cycle

    def buffer(self, max_bufsize: int) -> Iterator[bytes]:
        cycles_per_buffer = max_bufsize // len(self.cycle_data)
        if cycles_per_buffer <= 1:
            # Do not bother buffering, return the cycles as they are
            return chain(repeat(self.cycle_data, self.cycles), (self.last_cycle,))
        cycles_per_buffer = min(self.cycles, cycles_per_buffer)
        buffer_data = self.cycle_data * cycles_per_buffer
        buffer_cycles, extra_cycles = divmod(self.cycles, cycles_per_buffer)
        # We could also add the last cycle to the extra_cycles buffer
        # data, but that would add a lot more unnecessary code to check
        # that it stays under the specified maximum buffer size.
        extra_cycle_data = self.cycle_data * extra_cycles
        return chain(
            repeat(buffer_data, buffer_cycles), (extra_cycle_data, self.last_cycle)
        )


def _clamp_floats(floats: Iterable[float]):
    return (max(min(value, 1.0), -1.0) for value in floats)


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
    return (round((value - (-1.0)) * step_size + min_signed) for value in floats)


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
    return (
        amplitude * sin(angular_frequency * t / sample_rate) for t in range(num_samples)
    )


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
        Iterator[bytes]:
            An iterator of bytes objects containing PCM data.
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
    # "Exact cycle" algorithm
    # TODO: Research the correct terms to use. I'm not sure if "cycle"
    #       is the correct term to us here.
    # NOTE: `min_samples = sample_rate // gcd(frequency, sample_rate)``
    #       could also work, and in one step, but this two-step process
    #       helps to visualize the process. Keeping for readability.
    min_cycles = frequency // gcd(frequency, sample_rate)
    # Convert duration of exact cycle to duration in sample rate
    min_samples = sample_rate * min_cycles // frequency
    min_samples = min_samples if min_samples < num_samples else num_samples
    angular_frequency = 2 * pi * frequency
    exact_cycle = _generate_sine_wave(
        angular_frequency=angular_frequency,
        amplitude=amplitude,
        num_samples=min_samples,
        sample_rate=sample_rate,
    )
    if allow_clipping:
        exact_cycle = _clamp_floats(exact_cycle)
    exact_cycle = _map_floats_to_ints(exact_cycle, bits_per_sample=bits_per_sample)
    bytes_per_sample = bits_per_sample // 8
    exact_cycle = (
        i.to_bytes(bytes_per_sample, "little", signed=True) for i in exact_cycle
    )
    exact_cycle = (
        channel_data
        for data in exact_cycle
        for channel_data in repeat(data, num_channels)
    )
    exact_cycle_bytes = b"".join(exact_cycle)
    repetitions, last_cycle_samples = divmod(num_samples, min_samples)
    last_cycle_bytes = exact_cycle_bytes[:last_cycle_samples]
    # Append last_cycle_bytes onto the exact_cycle_bytes repetitions
    # TODO: Allow to get the last cycle separately for more efficient
    #       buffering when writing to files (repeat exact_cycle, join,
    #       then copy that joined buffer)
    output = PCMDataGenerator(exact_cycle_bytes, repetitions, last_cycle_bytes)
    return output


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
    """Writes the data for a PCM WAVE (.wav) file containing a sine
    wave.

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
    header = PCMWaveFileHeader(
        num_samples=num_samples,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
    ).to_bytes()
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
    for buffer in data.buffer(max_bufsize=buffer_size):
        bytes_written += fp.write(buffer)
    return bytes_written


# TODO: Add generate_silence
