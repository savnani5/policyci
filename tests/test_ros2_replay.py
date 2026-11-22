"""Bounded ROS 2 input replay with explicit soundness checks."""

from __future__ import annotations

import os
from pathlib import Path

from policyci.checkspec import CaseSpec, CheckSuite, GateSpec, RunnerSpec
from policyci.runner.ros2_replay import (
    CommandResult,
    Ros2ReplayRunner,
    SubprocessBackend,
)
from policyci.schemas import (
    ExecutionValidity,
    ReplayAnalysis,
    ReplayTrace,
    SignalSeries,
    SubjectRef,
)


class FakeProcess:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        exited: bool = False,
        stop_returncode: int = 0,
    ):
        self.name = name
        self.events = events
        self.exited = exited
        self.stop_returncode = stop_returncode

    def poll(self):
        return 1 if self.exited else None

    def stop(self, timeout_s: float):
        self.events.append(f"stop_{self.name}")
        return CommandResult(returncode=self.stop_returncode, stdout="", stderr="")


class FakeBackend:
    def __init__(
        self,
        *,
        topics: list[str] | None = None,
        readiness: list[int] | None = None,
        playback_returncode: int = 0,
        create_capture: bool = True,
        stale_capture: bool = False,
        subject_exit_after_playback: bool = False,
        recorder_exit_after_playback: bool = False,
        recorder_readiness: list[int] | None = None,
        create_metadata: bool = True,
        capture_counts: dict[str, int] | None = None,
        recorder_stop_returncode: int = 0,
    ):
        self.topics = topics or ["/vehicle/cmd_vel"]
        self.readiness = list(readiness or [0])
        self.playback_returncode = playback_returncode
        self.create_capture = create_capture
        self.stale_capture = stale_capture
        self.subject_exit_after_playback = subject_exit_after_playback
        self.recorder_exit_after_playback = recorder_exit_after_playback
        self.recorder_readiness = list(recorder_readiness or [0])
        self.create_metadata = create_metadata
        self.capture_counts = capture_counts or {"/user/cmd_vel": 1}
        self.recorder_stop_returncode = recorder_stop_returncode
        self.events: list[str] = []
        self.commands: list[list[str]] = []
        self.capture_dir: Path | None = None
        self.subject_process: FakeProcess | None = None
        self.recorder_process: FakeProcess | None = None

    def run(self, argv, *, timeout_s, env=None):
        self.commands.append(list(argv))
        if argv[:3] == ["ros2", "bag", "info"]:
            self.events.append("inspect_input")
            return CommandResult(
                returncode=0,
                stdout="\n".join(
                    f"Topic: {topic} | Type: geometry_msgs/msg/TwistStamped | Count: 1"
                    for topic in self.topics
                )
                + "\n",
                stderr="",
            )
        if argv and argv[0] == "probe":
            code = self.readiness.pop(0) if self.readiness else 1
            self.events.append("probe_ready" if code == 0 else "probe_not_ready")
            return CommandResult(returncode=code, stdout="", stderr="")
        if argv[:3] == ["ros2", "topic", "info"]:
            code = self.recorder_readiness.pop(0) if self.recorder_readiness else 1
            self.events.append(
                "probe_recorder_ready" if code == 0 else "probe_recorder_not_ready"
            )
            return CommandResult(
                returncode=code,
                stdout="Publisher count: 1\nSubscription count: 1\n" if code == 0 else "",
                stderr="",
            )
        if argv[:3] == ["ros2", "bag", "play"]:
            self.events.append("play_input")
            if self.create_capture and self.capture_dir is not None:
                self.capture_dir.mkdir(parents=True, exist_ok=True)
                output = self.capture_dir / "output.mcap"
                output.write_bytes(b"fresh-output")
                if self.create_metadata:
                    topics = "\n".join(
                        (
                            "    - topic_metadata:\n"
                            f"        name: {topic}\n"
                            "        type: geometry_msgs/msg/TwistStamped\n"
                            f"      message_count: {count}"
                        )
                        for topic, count in self.capture_counts.items()
                    )
                    total = sum(self.capture_counts.values())
                    (self.capture_dir / "metadata.yaml").write_text(
                        "rosbag2_bagfile_information:\n"
                        f"  message_count: {total}\n"
                        "  topics_with_message_count:\n"
                        f"{topics}\n"
                    )
                if self.stale_capture:
                    os.utime(output, (1, 1))
            if self.subject_exit_after_playback and self.subject_process is not None:
                self.subject_process.exited = True
            if self.recorder_exit_after_playback and self.recorder_process is not None:
                self.recorder_process.exited = True
            return CommandResult(
                returncode=self.playback_returncode,
                stdout="played",
                stderr="playback failed" if self.playback_returncode else "",
            )
        raise AssertionError(f"unexpected command: {argv}")

    def start(self, argv, *, env=None):
        self.commands.append(list(argv))
        if argv and argv[0] == "subject":
            self.events.append("start_subject")
            self.subject_process = FakeProcess("subject", self.events)
            return self.subject_process
        if argv[:3] == ["ros2", "bag", "record"]:
            self.events.append("start_recorder")
            self.capture_dir = Path(argv[argv.index("-o") + 1])
            self.recorder_process = FakeProcess(
                "recorder",
                self.events,
                stop_returncode=self.recorder_stop_returncode,
            )
            return self.recorder_process
        raise AssertionError(f"unexpected start command: {argv}")


class FakeAdapter:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        fail_analysis: bool = False,
        applied_fields: set[str] | None = None,
        analysis: dict | ReplayAnalysis | None = None,
    ):
        self.events = events
        self.fail_analysis = fail_analysis
        self.applied_fields = applied_fields
        self.analysis = analysis

    def applied_subject_fields(self, subject, case):
        del case
        if self.applied_fields is not None:
            return self.applied_fields
        fields = {
            name
            for name in ("checkpoint", "image", "code_sha")
            if getattr(subject, name) is not None
        }
        fields.update(f"config.{key}" for key in subject.config)
        return fields

    def subject_command(self, subject, case):
        return ["subject", subject.id]

    def readiness_command(self, subject, case):
        return ["probe", subject.id]

    def analyze(self, subject, case, capture):
        if self.events is not None:
            self.events.append("analyze_capture")
        if self.fail_analysis:
            raise ValueError("capture could not be decoded")
        return self.analysis or {"within_tolerance": True, "mean_abs_error": 0.01}


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 1.0
        return self.value


def _suite(
    bag: Path,
    *,
    repeats: int = 1,
    min_valid_repeats: int = 1,
    playback_topics: list[str] | None = None,
) -> CheckSuite:
    return CheckSuite(
        version=1,
        name="twist-replay",
        runner=RunnerSpec(
            type="ros2_replay",
            adapter="examples.ros2_replay.adapter:TwistReplayAdapter",
        ),
        gate=GateSpec(
            metric="within_tolerance",
            mode="require_value",
            expect=True,
            min_valid_repeats=min_valid_repeats,
        ),
        cases=[
            CaseSpec(
                id="twist",
                bag=bag,
                repeats=repeats,
                required_topics=["/vehicle/cmd_vel"],
                playback_topics=playback_topics or [],
                capture_topics=["/user/cmd_vel"],
            )
        ],
    )


def _runner(suite, backend, adapter=None, *, clock=None):
    return Ros2ReplayRunner(
        suite,
        adapter=adapter or FakeAdapter(backend.events),
        backend=backend,
        which=lambda executable: f"/usr/bin/{executable}",
        sleep=lambda seconds: None,
        monotonic=clock or Clock(),
    )


def test_preflight_rejects_missing_bag(tmp_path: Path):
    bag = tmp_path / "missing.mcap"
    backend = FakeBackend()
    result = _runner(_suite(bag), backend).preflight(
        SubjectRef(id="candidate", image="robot:latest"),
        _suite(bag).cases[0],
    )
    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "bag" in result.reasons[0]
    assert backend.events == []


def test_subprocess_environment_excludes_host_secrets(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("ROS_DISTRO", "jazzy")

    environment = SubprocessBackend._environment({"ROS_DOMAIN_ID": "41"})

    assert environment["ROS_DISTRO"] == "jazzy"
    assert environment["ROS_DOMAIN_ID"] == "41"
    assert "PATH" in environment
    assert {"GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY"}.isdisjoint(environment)


def test_preflight_rejects_declared_subject_identity_the_adapter_does_not_apply(
    tmp_path: Path,
):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag)
    runner = _runner(
        suite,
        backend,
        adapter=FakeAdapter(backend.events, applied_fields=set()),
    )

    result = runner.preflight(
        SubjectRef(id="candidate", image="robot:latest"),
        suite.cases[0],
    )

    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "image" in result.reasons[0]


def test_preflight_rejects_missing_required_topic(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(topics=["/other"])
    suite = _suite(bag)
    result = _runner(suite, backend).preflight(
        SubjectRef(id="candidate", image="robot:latest"), suite.cases[0]
    )
    assert result.validity is ExecutionValidity.NO_VERDICT
    assert "/vehicle/cmd_vel" in result.reasons[0]


def test_preflight_uses_rosbag_info_syntax_shared_by_humble_and_jazzy(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag)

    result = _runner(suite, backend).preflight(
        SubjectRef(id="candidate", image="robot:latest"), suite.cases[0]
    )

    assert result.validity is ExecutionValidity.VALID
    info = next(command for command in backend.commands if command[:3] == ["ros2", "bag", "info"])
    assert info == ["ros2", "bag", "info", str(bag)]


def test_playback_starts_only_after_subject_is_ready(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(readiness=[1, 0])
    suite = _suite(bag)
    runner = _runner(suite, backend)
    subject = SubjectRef(id="candidate", image="robot:latest")

    assert runner.preflight(subject, suite.cases[0]).validity is ExecutionValidity.VALID
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.VALID
    assert backend.events == [
        "inspect_input",
        "start_subject",
        "probe_not_ready",
        "probe_ready",
        "start_recorder",
        "probe_recorder_ready",
        "play_input",
        "stop_recorder",
        "stop_subject",
        "analyze_capture",
    ]


def test_playback_filters_to_required_inputs_not_capture_topics(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag)
    runner = _runner(suite, backend)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner.preflight(subject, suite.cases[0])
    runner.run_case(subject, suite.cases[0], tmp_path / "case")
    play = next(command for command in backend.commands if command[:3] == ["ros2", "bag", "play"])
    assert "/vehicle/cmd_vel" in play
    assert "/user/cmd_vel" not in play
    assert "--clock" in play
    assert play[play.index("--delay") + 1] == "1.0"
    assert play[play.index("--wait-for-all-acked") + 1] == "5000"
    record = next(
        command for command in backend.commands if command[:3] == ["ros2", "bag", "record"]
    )
    assert "--topics" in record
    assert record[record.index("--topics") + 1 :] == ["/user/cmd_vel"]


def test_playback_uses_the_suite_discovery_delay(tmp_path: Path) -> None:
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag)
    suite.execution.playback_discovery_delay_s = 5.0
    runner = _runner(suite, backend)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner.preflight(subject, suite.cases[0])

    runner.run_case(subject, suite.cases[0], tmp_path / "case")

    play = next(
        command
        for command in backend.commands
        if command[:3] == ["ros2", "bag", "play"]
    )
    assert play[play.index("--delay") + 1] == "5.0"


def test_playback_can_inject_only_the_component_input_subset(tmp_path: Path) -> None:
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(topics=["/camera", "/vehicle/cmd_vel"])
    suite = _suite(bag, playback_topics=["/vehicle/cmd_vel"])
    suite.cases[0].required_topics = ["/camera", "/vehicle/cmd_vel"]
    runner = _runner(suite, backend)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner.preflight(subject, suite.cases[0])

    runner.run_case(subject, suite.cases[0], tmp_path / "case")

    play = next(command for command in backend.commands if command[:3] == ["ros2", "bag", "play"])
    assert play[play.index("--topics") + 1 : play.index("--clock")] == [
        "/vehicle/cmd_vel"
    ]


def test_missing_fresh_capture_is_no_verdict(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(create_capture=False)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("capture" in reason for reason in execution.reasons)


def test_missing_recorder_metadata_is_no_verdict_even_when_metrics_exist(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(create_metadata=False)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("metadata" in reason for reason in execution.reasons)


def test_capture_topic_without_messages_is_no_verdict_even_when_adapter_passes(
    tmp_path: Path,
):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(capture_counts={"/other": 1})
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("/user/cmd_vel" in reason for reason in execution.reasons)


def test_recorder_finalization_failure_is_no_verdict(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(recorder_stop_returncode=7)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("finalization" in reason for reason in execution.reasons)


def test_stale_capture_is_no_verdict(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(stale_capture=True)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("fresh" in reason for reason in execution.reasons)


def test_adapter_metric_is_returned_in_case_execution(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")
    assert execution.metrics == {"within_tolerance": True, "mean_abs_error": 0.01}
    assert execution.unit_ids == ["repeat-000"]
    assert any(artifact.path.endswith("output.mcap") for artifact in execution.artifacts)
    assert all(artifact.kind != "trace" for artifact in execution.artifacts)


def test_runner_materializes_input_mcap_as_portable_repeat_artifact(
    tmp_path: Path,
) -> None:
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"immutable-input")
    backend = FakeBackend()
    suite = _suite(bag)
    runner = _runner(suite, backend)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    artifact = next(
        artifact for artifact in execution.artifacts if artifact.kind == "input_mcap"
    )
    copied = tmp_path / "case" / artifact.path
    assert artifact.path == "repeats/repeat-000/input.mcap"
    assert copied.read_bytes() == bag.read_bytes()


def test_runner_materializes_adapter_trace_as_portable_artifact(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag)
    analysis = ReplayAnalysis(
        metrics={"within_tolerance": False, "mean_abs_error": 0.4},
        trace=ReplayTrace(
            series={
                "expected": SignalSeries(values=[0.0], unit="m/s"),
                "observed": SignalSeries(values=[0.4], unit="m/s"),
                "absolute_error": SignalSeries(values=[0.4], unit="m/s"),
            }
        ),
    )
    runner = _runner(
        suite,
        backend,
        adapter=FakeAdapter(backend.events, analysis=analysis),
    )
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.VALID
    trace = next(artifact for artifact in execution.artifacts if artifact.kind == "trace")
    assert trace.path == "repeats/repeat-000/trace.json"
    stored = ReplayTrace.model_validate_json((tmp_path / "case" / trace.path).read_text())
    assert stored == analysis.trace


def test_readiness_timeout_prevents_recording_and_playback(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(readiness=[1, 1, 1, 1, 1])
    suite = _suite(bag)
    suite.execution.readiness_timeout_s = 2
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend, clock=Clock())
    runner.preflight(subject, suite.cases[0])
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("readiness timeout" in reason for reason in execution.reasons)
    assert "start_recorder" not in backend.events
    assert "play_input" not in backend.events
    assert "stop_subject" in backend.events


def test_playback_failure_still_stops_processes(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(playback_returncode=7)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("playback failed" in reason for reason in execution.reasons)
    assert "stop_recorder" in backend.events
    assert "stop_subject" in backend.events


def test_subject_crash_after_complete_capture_is_no_verdict(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(subject_exit_after_playback=True)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("subject exited during playback" in reason for reason in execution.reasons)


def test_recorder_crash_after_complete_capture_is_no_verdict(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend(recorder_exit_after_playback=True)
    suite = _suite(bag)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])

    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")

    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert any("recorder exited during playback" in reason for reason in execution.reasons)


def test_insufficient_valid_repeats_is_no_verdict_with_partial_artifacts(tmp_path: Path):
    bag = tmp_path / "input.mcap"
    bag.write_bytes(b"fixture")
    backend = FakeBackend()
    suite = _suite(bag, repeats=2, min_valid_repeats=2)
    subject = SubjectRef(id="candidate", image="robot:latest")
    runner = _runner(suite, backend)
    runner.preflight(subject, suite.cases[0])
    # The second repeat will not get a readiness success.
    backend.readiness = [0, 1, 1, 1]
    execution = runner.run_case(subject, suite.cases[0], tmp_path / "case")
    assert execution.validity is ExecutionValidity.NO_VERDICT
    assert execution.valid_units == 1
    assert execution.artifacts
    assert any("required=2" in reason for reason in execution.reasons)
