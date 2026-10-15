"""Round-trip NumPy dtype and shape metadata through Kamodo JSON helpers."""

import io
import json

import numpy as np
import pytest

from kamodo.util import deserialize, dump, dumps, load, loads, serialize


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "complex64",
        ">i4",
        ">f8",
        "bool",
        "U12",
    ],
)
def test_json_roundtrip_keeps_dtype_and_values(dtype):
    original = np.array([[0, 1], [1, 0]], dtype=dtype)
    encoded = dumps(original)
    restored = loads(encoded)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    np.testing.assert_array_equal(restored, original)
    assert json.loads(encoded)["dtype"] == original.dtype.str


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 3), (2, 3, 0), (0, 2, 0)])
def test_json_roundtrip_keeps_empty_dimensions(shape):
    original = np.empty(shape, dtype=np.float32)
    restored = loads(dumps(original))
    assert restored.shape == shape
    assert restored.dtype == original.dtype
    np.testing.assert_array_equal(restored, original)


@pytest.mark.parametrize(
    "source", [np.array(5, dtype="i2"), np.int16(7), np.float32(1.5), np.bool_(True)]
)
def test_scalar_payloads_remain_zero_dimensional_arrays(source):
    restored = deserialize(serialize(source))
    assert isinstance(restored, np.ndarray)
    assert restored.shape == ()
    assert restored.dtype == source.dtype
    np.testing.assert_array_equal(restored, source)


def test_file_helpers_preserve_nested_arrays_and_large_unsigned_integers():
    original = {
        "ids": np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64),
        "empty": np.empty((0, 4), dtype=np.int8),
    }
    stream = io.StringIO()
    dump(original, stream)
    stream.seek(0)
    restored = load(stream)
    for name, array in original.items():
        assert restored[name].dtype == array.dtype
        assert restored[name].shape == array.shape
        np.testing.assert_array_equal(restored[name], array)


@pytest.mark.parametrize(
    "source",
    [
        np.arange(24, dtype="i2").reshape(4, 6).T,
        np.arange(24, dtype="f4").reshape(4, 6)[:, ::2],
    ],
)
def test_noncontiguous_array_values_are_preserved(source):
    before = source.copy()
    restored = loads(dumps(source))
    assert restored.dtype == source.dtype
    assert restored.shape == source.shape
    np.testing.assert_array_equal(restored, source)
    np.testing.assert_array_equal(source, before)


def test_legacy_payload_without_metadata_still_infers_values():
    restored = deserialize({"__ndarray__": [[1, 2], [3, 4]]})
    np.testing.assert_array_equal(restored, [[1, 2], [3, 4]])
    assert restored.shape == (2, 2)


def test_legacy_payload_with_dtype_but_no_shape_is_supported():
    restored = deserialize({"__ndarray__": [1, 2], "dtype": "<i2"})
    assert restored.dtype == np.dtype("<i2")
    assert restored.shape == (2,)


def test_ordinary_json_objects_are_unchanged():
    original = {"values": [1, 2], "shape": [2], "dtype": "metadata"}
    assert loads(dumps(original)) == original


def test_invalid_shape_fails_rather_than_silently_changing_the_data():
    with pytest.raises(ValueError):
        deserialize({"__ndarray__": [1, 2], "dtype": "<i2", "shape": [3]})
