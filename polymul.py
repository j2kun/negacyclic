import math
import numpy
from numpy.fft import fft, ifft


def cyclic_polymul(p1, p2, N):
    """Multiply two polynomials modulo (x^N - 1).

    p1 and p2 are arrays of coefficients in degree-increasing order.
    """
    assert len(p1) == N
    assert len(p2) == N
    product = fft(p1) * fft(p2)
    inverted = ifft(product)
    return numpy.round(numpy.real(inverted)).astype(p1.dtype)


def negacyclic_polymul(p1, p2):
    """Multiply two polynomials modulo (x^N - 1).

    p1 and p2 are arrays of coefficients in degree-increasing order.
    """
    p1_preprocessed = numpy.concatenate([p1, -p1])
    p2_preprocessed = numpy.concatenate([p2, -p2])
    product = fft(p1_preprocessed) * fft(p2_preprocessed)
    inverted = ifft(product)
    rounded = numpy.round(0.5 * numpy.real(inverted)).astype(p1.dtype)
    return rounded[: p1.shape[0]]


def primitive_nth_root(n):
    """Return a primitive nth root of unity."""
    return math.cos(2 * math.pi / n) + 1.0j * math.sin(2 * math.pi / n)


def tangent_fft_negacyclic_polymul(p1, p2):
    """Computes a poly multiplication mod (X^N + 1) where N = len(a).

    Uses the Tangent FFT idea.
    """
    n = p2.shape[0]
    primitive_root = primitive_nth_root(2 * n)
    root_powers = primitive_root ** numpy.arange(n // 2)

    p1_preprocessed = (p1[: n // 2] + 1j * p1[n // 2 :]) * root_powers
    p2_preprocessed = (p2[: n // 2] + 1j * p2[n // 2 :]) * root_powers

    p1_ft = fft(p1_preprocessed)
    p2_ft = fft(p2_preprocessed)
    prod = p1_ft * p2_ft
    ifft_prod = ifft(prod)
    ifft_rotated = ifft_prod * primitive_root ** numpy.arange(0, -n // 2, -1)

    return numpy.round(
        numpy.concatenate([numpy.real(ifft_rotated), numpy.imag(ifft_rotated)])
    ).astype(p1.dtype)
