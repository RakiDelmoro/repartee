import pytest
from fastapi import HTTPException

from app.pipeline import parse_filename
from app.server import safe_filename


def test_accepts_the_device_filename():
    assert safe_filename("20261105-143012-rep01.wav") == "20261105-143012-rep01.wav"


def test_rejects_path_traversal():
    for bad in ["../x.wav", "a/b.wav", "..\\x.wav", "", None]:
        with pytest.raises(HTTPException):
            safe_filename(bad)


def test_rejects_wrong_extension_and_partial():
    for bad in ["notes.txt", "clip.wav.part", ".hidden.wav"]:
        with pytest.raises(HTTPException):
            safe_filename(bad)


def test_parses_device_and_start_time():
    assert parse_filename("20261105-143012-rep01.wav") == ("rep01", "2026-11-05T14:30:12")


def test_unparseable_names_still_get_a_device():
    assert parse_filename("mystery.wav") == ("mystery", None)
