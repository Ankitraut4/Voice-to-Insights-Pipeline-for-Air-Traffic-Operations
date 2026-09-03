from src.nlp_analysis import atlas, postprocess


CATEGORY_KEYWORDS = {
    "Emergency Declarations": ["emergency", "mayday"],
    "Miscellaneous": ["roger"],
}


def test_emergency_categorization_for_true_emergency_message():
    text = "mayday engine failure request priority landing"
    assert atlas.categorize_communication(text, CATEGORY_KEYWORDS) == "Emergency Declarations"
    assert postprocess.categorize_communication(text, CATEGORY_KEYWORDS) == "Emergency Declarations"


def test_non_emergency_message_with_emergency_token_is_not_misclassified_in_atlas():
    text = "good afternoon emergency 123"
    assert atlas.categorize_communication(text, CATEGORY_KEYWORDS) == "General Communications"


def test_postprocess_does_not_classify_ambiguous_emergency_reference():
    # Ambiguous use of "emergency" with flight-like token should not be auto-classified.
    text = "good afternoon emergency 123"
    assert postprocess.categorize_communication(text, CATEGORY_KEYWORDS) == "General Communications"
