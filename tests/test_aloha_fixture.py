from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from policyci.examples.aloha_replay.fixture import (
    AlohaSourceManifest,
    load_source_manifest,
    read_episode_parquet,
    recommended_gripper_tolerance,
    selected_source_paths,
    sha256_file,
    validate_episode_alignment,
)

PINNED_REVISION = "37fdb4dc78dc8a3b3f1fc44f7c7c60a207d18937"


def _manifest_payload() -> dict[str, object]:
    return {
        "repo_id": "lerobot/aloha_static_coffee",
        "revision": PINNED_REVISION,
        "episode_index": 0,
        "sample_rate_hz": 50,
        "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "task_label": "Make coffee",
        "license": "MIT",
        "source_url": "https://huggingface.co/datasets/lerobot/aloha_static_coffee",
    }


def test_source_manifest_accepts_only_immutable_revision() -> None:
    manifest = AlohaSourceManifest.model_validate(_manifest_payload())

    assert manifest.revision == PINNED_REVISION

    with pytest.raises(ValidationError, match="40-character"):
        AlohaSourceManifest.model_validate(_manifest_payload() | {"revision": "main"})


def test_checked_in_source_manifest_pins_the_per_episode_revision() -> None:
    manifest = load_source_manifest(Path("examples/aloha_replay/source.json"))

    assert manifest.revision == PINNED_REVISION
    assert manifest.episode_index == 0
    assert manifest.cameras == ("cam_high", "cam_left_wrist", "cam_right_wrist")


def test_selected_source_paths_are_one_episode_and_three_cameras() -> None:
    manifest = AlohaSourceManifest.model_validate(_manifest_payload())

    paths = selected_source_paths(manifest)

    assert paths == (
        "README.md",
        "meta/info.json",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.cam_high/episode_000000.mp4",
        "videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4",
        "videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4",
    )
    assert not any("cam_low" in path for path in paths)


def test_validate_episode_alignment_accepts_one_timestamp_per_camera_frame() -> None:
    validate_episode_alignment(
        sample_count=3,
        camera_frame_counts={"cam_high": 3, "cam_left_wrist": 3},
        timestamps_s=[0.0, 0.02, 0.04],
        sample_rate_hz=50.0,
    )


@pytest.mark.parametrize(
    ("camera_counts", "timestamps", "message"),
    [
        ({"cam_high": 2}, [0.0, 0.02, 0.04], "camera frame count"),
        ({"cam_high": 3}, [0.0, 0.02], "timestamp count"),
        ({"cam_high": 3}, [0.0, 0.02, 0.02], "strictly increasing"),
        ({"cam_high": 3}, [0.0, 0.03, 0.06], "50 Hz"),
    ],
)
def test_validate_episode_alignment_rejects_misalignment(
    camera_counts: dict[str, int],
    timestamps: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_episode_alignment(
            sample_count=3,
            camera_frame_counts=camera_counts,
            timestamps_s=timestamps,
            sample_rate_hz=50.0,
        )


def test_recommended_tolerance_is_ten_percent_of_smaller_gripper_range() -> None:
    actions = [[0.0] * 14 for _ in range(3)]
    actions[1][6] = 0.5
    actions[2][6] = 1.0
    actions[1][13] = 0.2
    actions[2][13] = 0.4

    assert recommended_gripper_tolerance(actions) == 0.04


def test_recommended_tolerance_rejects_constant_gripper_channel() -> None:
    with pytest.raises(ValueError, match="non-zero range"):
        recommended_gripper_tolerance([[0.0] * 14, [0.0] * 14])


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"aloha")

    assert sha256_file(source) == (
        "0206a97843b1ba4fbb147d472550ec3b5ee8aacadf3707522157240940d1bebd"
    )


def test_read_episode_parquet_loads_exact_robot_arrays(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "timestamp": [0.0, 0.02],
                "observation.state": [[0.0] * 14, [1.0] * 14],
                "observation.effort": [[2.0] * 14, [3.0] * 14],
                "action": [[4.0] * 14, [5.0] * 14],
            }
        ),
        source,
    )

    episode = read_episode_parquet(source)

    assert episode.timestamps_s == [0.0, 0.02]
    assert episode.states[1] == [1.0] * 14
    assert episode.efforts[0] == [2.0] * 14
    assert episode.actions[1] == [5.0] * 14


def test_read_episode_parquet_rejects_missing_effort(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "timestamp": [0.0],
                "observation.state": [[0.0] * 14],
                "action": [[0.0] * 14],
            }
        ),
        source,
    )

    with pytest.raises(ValueError, match="observation.effort"):
        read_episode_parquet(source)
