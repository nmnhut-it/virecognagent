"""
Quick smoke test — verifies all modules import and connect correctly.
Run: python scripts/test_smoke.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_viet_tools():
    """Test Vietnamese linguistic tools."""
    from src.viet_tools import (
        validate_vietnamese_text,
        is_valid_syllable_structure,
        check_tone_consistency,
        strip_tones,
        get_syllable_dictionary,
    )

    # Valid syllables
    assert is_valid_syllable_structure("xin"), "xin should be valid"
    assert is_valid_syllable_structure("chào"), "chào should be valid"
    assert is_valid_syllable_structure("việt"), "việt should be valid"
    assert is_valid_syllable_structure("nam"), "nam should be valid"
    assert is_valid_syllable_structure("nghiệp"), "nghiệp should be valid"

    # Tone stripping
    assert strip_tones("chào") == "chao", f"Expected 'chao', got '{strip_tones('chào')}'"
    assert strip_tones("Việt") == "viêt", f"Tone strip failed: got '{strip_tones('Việt')}'"

    # Tone consistency
    valid, info = check_tone_consistency("chào")
    assert valid, "chào should have valid tone"

    # Full validation
    result = validate_vietnamese_text("Xin chào Việt Nam")
    assert result.confidence > 0.5, f"Confidence too low: {result.confidence}"

    # Dictionary size
    d = get_syllable_dictionary()
    assert len(d) > 1000, f"Dictionary too small: {len(d)}"

    print("  ✓ viet_tools: all tests passed")


def test_metrics():
    """Test CER, WER, DER computation."""
    from src.metrics import evaluate, compute_der, compute_cer

    # Perfect match
    m = evaluate(["Xin chào"], ["Xin chào"])
    assert m.cer == 0.0, f"CER should be 0, got {m.cer}"
    assert m.der == 0.0, f"DER should be 0, got {m.der}"

    # Stripped diacritics
    m = evaluate(["Xin chao Viet Nam"], ["Xin chào Việt Nam"])
    assert m.cer > 0, "CER should be > 0"
    assert m.der > 0, "DER should be > 0"
    assert m.n_diacritic_chars > 0, "Should have diacritic chars"

    # DER should be >= CER (diacritics are a subset of all chars)
    # Actually DER can differ since it's computed on different char sets

    # Tone error rate
    assert m.tone_er >= 0, "Tone ER should be >= 0"

    print("  ✓ metrics: all tests passed")


def test_structured_output():
    """Test structured output classes."""
    from src.pipeline import (
        RegionOutput, StructuredOutput, DiacriticInfo, AgentTrace
    )
    import json

    # Create a region
    region = RegionOutput(
        id=0,
        bbox=[[0, 0], [100, 0], [100, 30], [0, 30]],
        text="Xin chào",
        confidence=0.92,
        method="vietocr_high_conf",
        agent_activated=False,
        diacritics=DiacriticInfo(count=1, chars=["à"], corrections_made=0, der_risk="low"),
    )

    d = region.to_dict()
    assert d["text"] == "Xin chào"
    assert d["confidence"] == 0.92
    assert d["diacritics"]["count"] == 1

    # Create a region with agent trace
    region_agent = RegionOutput(
        id=1,
        bbox=[[0, 30], [100, 30], [100, 60], [0, 60]],
        text="Việt Nam",
        confidence=0.61,
        method="agent_pod_r2",
        agent_activated=True,
        agent_trace=AgentTrace(
            initial_text="Viet Nam",
            proposer_text="Việt Nam",
            linguist_verdict="CORRECTED",
            judge_text="Việt Nam",
            rounds=2,
            safety_rejected=False,
        ),
        diacritics=DiacriticInfo(count=2, chars=["ệ"], corrections_made=1, der_risk="medium"),
    )

    d2 = region_agent.to_dict()
    assert d2["agent_activated"] is True
    assert d2["agent_trace"]["rounds"] == 2

    # Create full structured output
    output = StructuredOutput(
        document="test.png",
        regions=[region, region_agent],
        full_text="Xin chào\nViệt Nam",
        metadata={"total_regions": 2, "agent_activation_rate": 0.5},
    )

    json_str = output.to_json()
    parsed = json.loads(json_str)
    assert len(parsed["regions"]) == 2
    assert parsed["metadata"]["agent_activation_rate"] == 0.5

    print("  ✓ structured_output: all tests passed")


def test_pipeline_modes():
    """Test that all pipeline modes can be referenced."""
    from src.pipeline import ViRecognAgentPipeline

    expected_modes = [
        "gemma4_zeroshot", "gemma4_think", "gemma4_comprehensive",
        "gemma4_cot", "gemma4_sc", "vietocr", "vietocr_ensemble",
        "two_stage", "full", "ablation_no_linguist",
        "ablation_no_judge", "ablation_tools_only",
    ]

    for mode in expected_modes:
        assert mode in ViRecognAgentPipeline.MODES, f"Missing mode: {mode}"

    print(f"  ✓ pipeline_modes: all {len(expected_modes)} modes registered")


def test_calibration():
    """Test confidence calibration module."""
    from src.calibration import compute_calibration, is_correct_match

    confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85, 0.3, 0.2, 0.1]
    is_correct = [True, True, True, False, False, True, True, False, False, False]

    result = compute_calibration(confidences, is_correct, n_bins=5, threshold=0.7)
    assert 0 <= result.ece <= 1, f"ECE out of range: {result.ece}"
    assert 0 <= result.mce <= 1, f"MCE out of range: {result.mce}"
    assert 0 <= result.optimal_threshold <= 1, f"Threshold out of range"
    assert result.n_samples == 10

    # Test correct match function
    assert is_correct_match("hello", "hello") is True
    assert is_correct_match("hello", "world") is False
    assert is_correct_match("Xin chào", "Xin chao") is False  # CER > 5%

    print("  ✓ calibration: all tests passed")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("ViRecognAgent Smoke Tests")
    print("=" * 50 + "\n")

    tests = [
        ("Vietnamese Tools", test_viet_tools),
        ("Metrics (CER/WER/DER)", test_metrics),
        ("Structured Output", test_structured_output),
        ("Pipeline Modes", test_pipeline_modes),
        ("Calibration", test_calibration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}\n")

    if failed > 0:
        sys.exit(1)
