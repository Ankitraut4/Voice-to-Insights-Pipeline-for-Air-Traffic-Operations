from src.nlp_analysis import atlas, postprocess


def test_preprocess_transcript_handles_repetition_and_punctuation():
    text = "thank you. thank you. thank you."
    expected = "thank you."

    assert atlas.preprocess_transcript(text) == expected
    assert postprocess.preprocess_transcript(text) == expected
    assert atlas.preprocess_transcript("hello hello") == "hello"
    assert postprocess.preprocess_transcript("hello hello") == "hello"


def test_spoken_number_normalization_matches_current_behavior():
    text = "heading two seven zero maintain altitude five thousand"
    expected = "heading 2 7 0 maintain altitude 5 000"

    assert atlas.convert_spoken_numbers(text) == expected
    assert postprocess.convert_spoken_numbers(text) == expected


def test_invalid_low_quality_filtering_common_cases():
    assert atlas.is_valid_transcription(".....") is False
    assert postprocess.is_valid_transcription(".....") is False
    assert atlas.is_valid_transcription("normal communication") is True
    assert postprocess.is_valid_transcription("normal communication") is True


def test_gratitude_only_transcription_is_rejected_consistently():
    assert atlas.is_valid_transcription("thank you captain") is False
    assert postprocess.is_valid_transcription("thank you captain") is False


def test_duplicate_flagging_marks_both_original_and_repeat():
    items = [
        {"raw_transcription": "hello hello"},
        {"raw_transcription": "hello"},
        {"raw_transcription": "different"},
    ]

    flagged, duplicate_count = atlas.flag_duplicates(items)

    assert duplicate_count == 1
    assert flagged[0]["duplicate_flag"] is True
    assert flagged[1]["duplicate_flag"] is True
    assert flagged[2]["duplicate_flag"] is False
