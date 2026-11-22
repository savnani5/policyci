from pathlib import Path


def test_live_aloha_script_is_offline_and_verifies_the_expected_gate() -> None:
    script = Path("scripts/run_aloha_replay_demo.sh").read_text()

    assert "curl " not in script
    assert "huggingface" not in script.lower()
    assert "--network none" in script
    assert 'OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"' in script
    assert 'CHECK_EXIT}" -ne 1' in script
    assert 'comparison["status"] != "regressed"' in script
    assert 'comparison["baseline_value"] is not True' in script
    assert 'comparison["candidate_value"] is not False' in script
    assert '{"input_mcap", "mcap", "trace"}' in script
    assert "rerun rrd verify" in script
    assert "aloha_coffee_replay" in script
