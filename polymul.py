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
    return numpy.round(numpy.real(inverted)).astype(numpy.int32)
