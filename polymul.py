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


def negacyclic_polymul_preimage_and_map_back(p1, p2):
    """Multiply two polynomials modulo (x^N + 1).

    p1 and p2 are arrays of coefficients in degree-increasing order.

    This is the technique that maps the polynomials to a larger ring
    R[x]/(x^2N - 1), but does not take advantage of the extra structure in the
    preimage ring, and instead maps it back down to the smaller ring manually.
    """
    p1_preprocessed = numpy.concatenate([p1, -p1])
    p2_preprocessed = numpy.concatenate([p2, -p2])
    product = fft(p1_preprocessed) * fft(p2_preprocessed)
    inverted = ifft(product)
    rounded = numpy.round(numpy.real(inverted)).astype(p1.dtype)
    return (rounded[: p1.shape[0]] - rounded[p1.shape[0] :]) // 4


def wrapping_convolve(a, b):
    """Implement a wrapping convolution using numpy.convolve.

    Numpy doesn't have a wrapping option, so this requires repeating
    the coefficients of at least one of the input arrays.

    Also note that, because the "valid" mode computes a convolution only when the
    windows fully overlap, the first and last values are the same and should
    correspond to the very last value in the output. Hence the final slicing step.
    """
    conv_result = numpy.convolve(a, numpy.tile(b, 2), mode="valid")
    return conv_result[1:]


def negacyclic_polymul_preimage_and_map_back_conv(p1, p2):
    """Multiply two polynomials modulo (x^N + 1).

    Same as negacyclic_polymul_preimage_and_map_back,
    but uses a convolution operation instead of an FFT.
    """
    p1_preprocessed = numpy.concatenate([p1, -p1])
    p2_preprocessed = numpy.concatenate([p2, -p2])
    product = wrapping_convolve(p1_preprocessed, p2_preprocessed)
    return (product[: p1.shape[0]] - product[p1.shape[0] :]) // 4


def negacyclic_polymul_use_special_preimage(p1, p2):
    """Multiply two polynomials modulo (x^N + 1).

    p1 and p2 are arrays of coefficients in degree-increasing order.

    This is the technique that maps the polynomials to a larger ring
    R[x]/(x^2N - 1), and uses the extra structure in the preimage ring to avoid
    having to manually map it back to the smaller ring.
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


def negacyclic_polymul_complex_twist(p1, p2):
    """Computes a poly multiplication mod (X^N + 1) where N = len(p1).

    Uses the idea on page 332 (pdf page 8) of "Fast multiplication and its
    applications" by Daniel Bernstein.
    http://cr.yp.to/lineartime/multapps-20080515.pdf
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


def cylic_matrix(c: numpy.array) -> numpy.ndarray:
    """Generates a cyclic matrix with each row of the input shifted.

    For input: [1, 2, 3], generates the following matrix:

        [[1 2 3]
         [2 3 1]
         [3 1 2]]
    """
    c = numpy.asarray(c).ravel()
    a, b = numpy.ogrid[0 : len(c), 0 : -len(c) : -1]
    indx = a + b
    return c[indx]


def negacyclic_polymul_toeplitz(p1, p2):
    """Computes a poly multiplication mod (X^N + 1) where N = len(p1).

    Uses the Toeplitz matrix representation of p1 to convert the problem into a
    vector-matrix product.
    """
    n = len(p1)

    # Generates a sign matrix with 1s above the diagonal and -1 above.
    up_tri = numpy.tril(numpy.ones((n, n), dtype=int), 0)
    low_tri = numpy.triu(numpy.ones((n, n), dtype=int), 1) * -1
    sign_matrix = up_tri + low_tri

    cyclic_matrix = cylic_matrix(p1)
    toeplitz_p1 = sign_matrix * cyclic_matrix
    return numpy.matmul(toeplitz_p1, p2)
