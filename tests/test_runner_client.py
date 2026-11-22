"""Modal fanout artifact materialization."""

from pathlib import Path

from policyci.runner import client


def test_fanout_pulls_partial_evidence_from_failed_shards(
    tmp_path: Path,
    monkeypatch,
):
    pulled: list[str] = []

    def fake_pull(run_id: str, root: Path, *, use_existing_app: bool):
        assert root == tmp_path
        assert use_existing_app is False
        pulled.append(run_id)

    monkeypatch.setattr(client, "pull_volume_run", fake_pull)

    client._pull_fanout_runs(
        [
            {"run_id": "aggregate/shard_0000", "status": "succeeded"},
            {"run_id": "aggregate/shard_0001", "status": "failed"},
            {"status": "failed"},
        ],
        tmp_path,
    )

    assert pulled == ["aggregate/shard_0000", "aggregate/shard_0001"]
