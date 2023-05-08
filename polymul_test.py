from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
import numpy as np

import polymul


def _np_polymul_mod(poly1, poly2, poly_mod):
    # Reversing the list order because numpy polymul interprets the polynomial
    # with higher-order coefficients first, whereas our code does the opposite
    np_mul = np.polymul(list(reversed(poly1)), list(reversed(poly2)))
    (_, np_poly_mod) = np.polydiv(np_mul, poly_mod)
    np_pad = np.pad(
        np_poly_mod,
        (len(poly1) - len(np_poly_mod), 0),
        "constant",
        constant_values=(0, 0),
    )
    return np.array(list(reversed(np_pad)), dtype=np.int32)


def _np_negacyclic_polymul(poly1, poly2):
    # a reference implementation for negacyclic polynomial multiplication
    # poly_mod represents the polynomial to divide by: x^N + 1, N = len(a)
    poly_mod = np.zeros(len(poly1) + 1, np.uint32)
    poly_mod[0] = 1
    poly_mod[len(poly1)] = 1
    return _np_polymul_mod(poly1, poly2, poly_mod)


def _np_cyclic_polymul(poly1, poly2):
    # a reference implementation of cyclic polynomial multiplication
    # poly_mod represents the polynomial to divide by: x^N - 1, N = len(a)
    poly_mod = np.zeros(len(poly1) + 1, np.uint32)
    poly_mod[0] = 1
    poly_mod[len(poly1)] = -1
    return _np_polymul_mod(poly1, poly2, poly_mod)


N = 16


@given(
    lists(integers(min_value=-100, max_value=100), min_size=N, max_size=N),
    lists(integers(min_value=-100, max_value=100), min_size=N, max_size=N),
)
def test_cyclic_polymul(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    expected = _np_cyclic_polymul(p1, p2)
    actual = polymul.cyclic_polymul(p1, p2, len(p1))
    np.testing.assert_array_equal(expected, actual)


NEGACYCLIC_IMPLS = [
    polymul.negacyclic_polymul_preimage_and_map_back_conv,
    polymul.negacyclic_polymul_preimage_and_map_back,
    polymul.negacyclic_polymul_use_special_preimage,
    polymul.negacyclic_polymul_complex_twist,
    polymul.negacyclic_polymul_toeplitz,
]


@given(
    lists(integers(min_value=-100, max_value=100), min_size=N, max_size=N),
    lists(integers(min_value=-100, max_value=100), min_size=N, max_size=N),
    sampled_from(NEGACYCLIC_IMPLS),
)
def test_negacyclic_polymul(p1, p2, impl):
    p1 = np.array(p1)
    p2 = np.array(p2)
    expected = _np_negacyclic_polymul(p1, p2)
    if impl == polymul.negacyclic_polymul_preimage_and_map_back_conv:
        print()
        print(f"Multiplying {p1} with {p2}")
        print(f"Expecting {expected}")
    actual = impl(p1, p2)
    if impl == polymul.negacyclic_polymul_preimage_and_map_back_conv:
        print(f"Actual {actual}")
    np.testing.assert_array_equal(expected, actual)
