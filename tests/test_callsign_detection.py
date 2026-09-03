from pathlib import Path

from src.nlp_analysis import atlas, postprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALLSIGN_PATH = PROJECT_ROOT / "config" / "airline_callsign.json"
PHONETIC_PATH = PROJECT_ROOT / "config" / "phonetic_alphabet.json"


def test_commercial_callsign_detection_works_with_canonical_config():
    callsigns = atlas.load_callsigns(CALLSIGN_PATH)
    detected = atlas.detect_callsign("delta one two three request descent", callsigns)
    assert detected == "Delta Air Lines"


def test_postprocess_commercial_callsign_detection_works_with_canonical_config():
    callsigns = postprocess.load_callsigns(CALLSIGN_PATH)
    detected = postprocess.detect_callsign("delta one two three request descent", callsigns)
    assert detected == "Delta Air Lines"


def test_general_aviation_direct_n_number_detection():
    callsigns = atlas.load_callsigns(CALLSIGN_PATH)
    phonetic = atlas.load_phonetic_alphabet(PHONETIC_PATH)
    detected = atlas.detect_callsign("N123AB request landing", callsigns, phonetic)
    assert detected == "General Aviation (N123AB)"


def test_postprocess_general_aviation_direct_n_number_detection():
    callsigns = postprocess.load_callsigns(CALLSIGN_PATH)
    detected = postprocess.detect_callsign("N123AB request landing", callsigns)
    assert detected == "General Aviation (N123AB)"


def test_general_aviation_november_phonetic_detection():
    callsigns = atlas.load_callsigns(CALLSIGN_PATH)
    phonetic = atlas.load_phonetic_alphabet(PHONETIC_PATH)
    detected = atlas.detect_callsign("november one two three alpha bravo", callsigns, phonetic)
    assert detected == "General Aviation (N123AB)"
