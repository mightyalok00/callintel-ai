"""
Unit tests for Speech Benchmark metrics (WER & DER).
Verifies standard Levenshtein word error rate and NIST collar diarization calculations.
"""

from training.benchmark_speech import compute_wer, compute_der, normalize_words


def test_wer_identical_strings():
    ref = "Hello, thank you for calling customer service."
    hyp = "Hello thank you for calling customer service"
    wer, details = compute_wer(ref, hyp)
    assert wer == 0.0
    assert details["substitutions"] == 0
    assert details["deletions"] == 0
    assert details["insertions"] == 0


def test_wer_substitutions_and_insertions():
    ref = "I want to cancel my subscription today"
    hyp = "I want to terminate my subscription please today"
    # 'cancel' -> 'terminate' (1 sub), 'please' inserted (1 ins)
    wer, details = compute_wer(ref, hyp)
    assert wer > 0.0
    assert details["substitutions"] == 1
    assert details["insertions"] == 1


def test_der_perfect_alignment():
    ref_turns = [
        {"speaker": "Agent", "text": "Hello how can I help you today?"},
        {"speaker": "Customer", "text": "I want to discuss my account billing."},
    ]
    hyp_turns = [
        {
            "speaker": "Speaker 0",
            "utterance": "Hello how can I help you today?",
            "start_seconds": 0.0,
            "end_seconds": 4.0,
        },
        {
            "speaker": "Speaker 1",
            "utterance": "I want to discuss my account billing.",
            "start_seconds": 4.5,
            "end_seconds": 9.0,
        },
    ]

    metrics = compute_der(ref_turns, hyp_turns, collar_seconds=0.25)
    assert metrics["der"] == 0.0
    assert metrics["speaker_error_rate"] == 0.0
    assert metrics["mapped_speakers"]["Speaker 0"] == "Agent"
    assert metrics["mapped_speakers"]["Speaker 1"] == "Customer"


def test_der_speaker_confusion():
    ref_turns = [
        {"speaker": "Agent", "text": "Hello how can I help you today?"},
        {"speaker": "Customer", "text": "I want to discuss my account billing."},
    ]
    # Inverted or single cluster assignment
    hyp_turns = [
        {
            "speaker": "Speaker 0",
            "utterance": "Hello how can I help you today?",
            "start_seconds": 0.0,
            "end_seconds": 4.0,
        },
        {
            "speaker": "Speaker 0",  # Both assigned to Speaker 0 -> confusion on turn 2
            "utterance": "I want to discuss my account billing.",
            "start_seconds": 4.5,
            "end_seconds": 9.0,
        },
    ]

    metrics = compute_der(ref_turns, hyp_turns, collar_seconds=0.25)
    assert metrics["der"] > 0.0
