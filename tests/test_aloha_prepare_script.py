from pathlib import Path


def test_prepare_script_builds_pinned_image_and_mounts_only_required_paths() -> None:
    script = Path("scripts/prepare_aloha_replay_fixture.sh").read_text()

    assert "examples/ros2_replay/Dockerfile" in script
    assert "policyci.examples.aloha_replay.fixture" in script
    assert "source.json:/workspace/source.json:ro" in script
    assert '--volume "${FIXTURE_DIR}:/fixtures"' in script
    assert '--volume "${CACHE_DIR}:/cache"' in script
    assert "aloha_coffee_episode_000000.mcap" in script
