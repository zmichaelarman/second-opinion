import json
import unicodedata

import pytest

from second_opinion.fixtures.payloads import EXPECTED_ARGS, MARKUP_STRING
from second_opinion.oracles import typed_equal


def codes(expected, got):
    return [m.code for m in typed_equal(expected, got)]


def notes(expected, got):
    return [m.note for m in typed_equal(expected, got)]


@pytest.mark.parametrize("variant", sorted(EXPECTED_ARGS))
def test_round_tripped_expected_args_are_equal(variant):
    args = EXPECTED_ARGS[variant]
    assert typed_equal(args, json.loads(json.dumps(args, ensure_ascii=False))) == []


def test_leading_zero_string_to_int_is_coerced():
    assert codes({"zip_code": "02134"}, {"zip_code": 2134}) == ["TYPE_COERCED"]
    assert notes({"zip_code": "02134"}, {"zip_code": 2134}) == ["STRING_TO_NUMBER"]


def test_int_to_string_is_coerced():
    assert codes({"street_number": 42}, {"street_number": "42"}) == ["TYPE_COERCED"]


def test_int_to_float_is_coerced():
    assert notes({"street_number": 42}, {"street_number": 42.0}) == ["INT_TO_FLOAT"]


@pytest.mark.parametrize("got", [9007199254740992, 9007199254740994, 9007199254740992.0])
def test_big_int_precision_loss(got):
    assert codes({"big_id": 9007199254740993}, {"big_id": got}) == ["TYPE_COERCED"]
    assert notes({"big_id": 9007199254740993}, {"big_id": got}) == ["PRECISION_LOSS"]


def test_big_int_far_off_is_a_mismatch():
    assert codes({"big_id": 9007199254740993}, {"big_id": 9007199254740000}) == ["VALUE_MISMATCH"]


def test_bool_is_never_equal_to_int():
    assert codes({"is_po_box": False}, {"is_po_box": 0}) == ["TYPE_COERCED"]
    assert codes({"n": 1}, {"n": True}) == ["TYPE_COERCED"]
    assert codes({"is_po_box": False}, {"is_po_box": 5}) == ["VALUE_MISMATCH"]


def test_null_and_bool_as_strings_are_coerced():
    assert codes({"unit": None}, {"unit": "null"}) == ["TYPE_COERCED"]
    assert codes({"is_po_box": False}, {"is_po_box": "false"}) == ["TYPE_COERCED"]


def test_nested_object_sent_as_json_string_is_coerced():
    exp = {"geo": {"lat": 42.35, "lon": -71.13}}
    assert notes(exp, {"geo": json.dumps(exp["geo"])}) == ["JSON_ENCODED_STRING"]


def test_markup_cut_before_closing_parameter_tag_is_truncated():
    cut_with_newline = MARKUP_STRING[: MARKUP_STRING.index("</parameter>")]
    cut_stripped = cut_with_newline.rstrip("\n")
    for got in (cut_with_newline, cut_stripped):
        assert codes({"content": MARKUP_STRING}, {"content": got}) == ["VALUE_TRUNCATED"]


def test_markup_cut_before_less_than_sign_is_truncated():
    got = MARKUP_STRING[: MARKUP_STRING.index("< 6")]
    assert codes({"content": MARKUP_STRING}, {"content": got}) == ["VALUE_TRUNCATED"]


def test_other_string_change_is_a_mismatch_with_diff():
    [m] = typed_equal({"content": MARKUP_STRING}, {"content": MARKUP_STRING.replace("naïve", "naive")})
    assert m.code == "VALUE_MISMATCH" and "-He said" in m.note and "+He said" in m.note


def test_unicode_normalization_difference_is_noted():
    [m] = typed_equal({"content": MARKUP_STRING}, {"content": unicodedata.normalize("NFD", MARKUP_STRING)})
    assert m.code == "VALUE_MISMATCH" and "Unicode normalization" in m.note


def test_missing_extra_keys_and_lengths_are_mismatches():
    assert codes({"a": 1, "b": 2}, {"a": 1}) == ["VALUE_MISMATCH"]
    assert codes({"a": 1}, {"a": 1, "z": 0}) == ["VALUE_MISMATCH"]
    assert codes({"tags": ["a", "b"]}, {"tags": ["a"]}) == ["VALUE_MISMATCH"]


def test_paths_point_at_the_nested_field():
    [m] = typed_equal(EXPECTED_ARGS["bool_null_nested"], {**EXPECTED_ARGS["bool_null_nested"], "geo": {"lat": "42.35", "lon": -71.13}})
    assert (m.code, m.path, m.note) == ("TYPE_COERCED", "$.geo.lat", "NUMBER_TO_STRING")


def test_several_problems_are_all_reported():
    got = {"zip_code": 2134, "account_id": "7", "numeric_label": "12345"}
    assert sorted(codes(EXPECTED_ARGS["leading_zero"], got)) == ["TYPE_COERCED", "VALUE_MISMATCH"]
