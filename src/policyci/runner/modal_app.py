"""Modal app: LeRobot/LIBERO eval on cloud GPUs.

v0: single-worker eval_checkpoint
v1: .map() fan-out over scenario shards
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "policyci"
VOLUME_NAME = "policyci-runs"
VOLUME_MOUNT = "/vol/runs"

# Pin lerobot to a release compatible with pi05_libero_finetuned_v044.
# Adjust if Hub model card specifies a different pin.
# Extra is `pi` (Pi0/Pi0.5), not `pi0`. Pin matches Hub `_v044` checkpoints.
# NOTE: lerobot[libero] and lerobot[pi] conflict on transformers (stock vs
# fix/lerobot_openpi branch). Install base + hf-libero, then patched transformers.
LEROBOT_VERSION = "0.4.4"
TRANSFORMERS_OPENPI = (
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
)

app = modal.App(APP_NAME)

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Headless EGL sim image — cached once, reused across workers.
sim_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "cmake",
        "build-essential",
        "clang",
        "linux-libc-dev",
        "libgl1",
        "libglx-mesa0",
        "libegl1",
        "libglew-dev",
        "libosmesa6-dev",
        "libglfw3",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "mesa-utils",
        "libglvnd0",
        "libgl1-mesa-glx",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        }
    )
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Do NOT install stock transformers here — PI05 needs the openpi fork.
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "gymnasium",
        "imageio[ffmpeg]",
        "opencv-python-headless",
        "numpy",
        "einops",
        "draccus",
        "rerun-sdk",
        "termcolor",
        "deepdiff",
        "jsonlines",
        "pydantic",
        "packaging",
        "av",
        "diffusers",
        "evdev",
        "pyyaml",
        "scipy>=1.10.1,<1.15",
        "num2words",  # required by SmolVLM processor (smolvla policies)
    )
    # Base lerobot + LIBERO env (without transformers-dep conflict)
    .pip_install(f"lerobot=={LEROBOT_VERSION}", "hf-libero>=0.1.3,<0.2.0")
    # Patched transformers required by PI0/PI05 (provides transformers.models.siglip.check)
    .pip_install(TRANSFORMERS_OPENPI)
    # Reinstall CUDA torch last (pip deps may pull CPU wheels)
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import yaml\n"
        "import importlib.util\n"
        "spec = importlib.util.find_spec('libero')\n"
        "assert spec and spec.origin\n"
        "pkg = Path(spec.origin).resolve().parent\n"
        "root = pkg / 'libero' if (pkg / 'libero' / 'bddl_files').exists() else pkg\n"
        "if not (root / 'bddl_files').exists() and spec.submodule_search_locations:\n"
        "    for p in [pkg, pkg / 'libero', Path(list(spec.submodule_search_locations)[0]) / 'libero']:\n"
        "        if (p / 'bddl_files').exists():\n"
        "            root = p\n"
        "            break\n"
        "cfg_dir = Path.home() / '.libero'\n"
        "cfg_dir.mkdir(parents=True, exist_ok=True)\n"
        "assets = Path.home() / '.cache' / 'libero' / 'assets'\n"
        "assets.mkdir(parents=True, exist_ok=True)\n"
        "cfg = {\n"
        "  'benchmark_root': str(root),\n"
        "  'bddl_files': str(root / 'bddl_files'),\n"
        "  'init_states': str(root / 'init_files'),\n"
        "  'datasets': str(root.parent / 'datasets'),\n"
        "  'assets': str(assets),\n"
        "}\n"
        "(cfg_dir / 'config.yaml').write_text(yaml.dump(cfg))\n"
        "print('libero config', cfg)\n"
        "PY",
        # Verify the openpi transformers patch is present (no GPU needed)
        "python -c \"from transformers.models.siglip import check; "
        "assert check.check_whether_transformers_replace_is_installed_correctly(); "
        "print('transformers openpi patch OK'); import libero.libero; print('libero ok')\"",
    )
)

# Back-compat alias used in older comments/docs
LEROBOT_PIN = f"lerobot[libero,pi]=={LEROBOT_VERSION}"


hf_secret = modal.Secret.from_name("huggingface-secret")


def _parse_eval_info(
    eval_info: dict[str, Any],
    *,
    suite: str,
    policy_id: str,
    seed: int,
) -> dict[str, Any]:
    """Normalize LeRobot eval_info into tasks + episodes lists.

    Handles:
    - flat single-task: {per_episode, aggregated, video_paths}
    - nested dict per_group/per_task: {suite: {task_id: metrics}}
    - list per_task (lerobot 0.4.4): [{task_group, task_id, metrics: {successes, ...}}]
    """
    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    # List-style per_task (observed on Modal derisk with lerobot 0.4.4)
    if isinstance(eval_info.get("per_task"), list):
        for entry in eval_info["per_task"]:
            if not isinstance(entry, dict):
                continue
            group = str(entry.get("task_group", suite))
            tid = entry.get("task_id", 0)
            metrics = entry.get("metrics") or {}
            successes = list(metrics.get("successes") or [])
            n_ep = len(successes) or int(metrics.get("n_episodes", 0) or 0)
            n_ok = sum(1 for s in successes if s)
            pc = (100.0 * n_ok / n_ep) if n_ep else float(metrics.get("pc_success", 0) or 0)
            video_paths = list(metrics.get("video_paths") or [])
            seeds = [seed + i for i in range(n_ep)]
            tasks.append(
                {
                    "task": f"{group}/{tid}",
                    "task_id": int(tid) if tid is not None and str(tid).isdigit() else tid,
                    "suite": group,
                    "n_episodes": n_ep,
                    "n_successes": n_ok,
                    "pc_success": pc,
                    "seeds": seeds,
                    "video_paths": video_paths,
                }
            )
            for i, ok in enumerate(successes):
                episodes.append(
                    {
                        "episode_id": f"{group}_{tid}_ep{i}",
                        "task": f"{group}/{tid}",
                        "task_id": int(tid) if tid is not None and str(tid).isdigit() else None,
                        "suite": group,
                        "instruction": "",
                        "seed": seeds[i],
                        "success": bool(ok),
                        "policy_id": policy_id,
                        "video_path": video_paths[i] if i < len(video_paths) else None,
                        "sum_reward": (metrics.get("sum_rewards") or [None])[i]
                        if i < len(metrics.get("sum_rewards") or [])
                        else None,
                        "max_reward": (metrics.get("max_rewards") or [None])[i]
                        if i < len(metrics.get("max_rewards") or [])
                        else None,
                    }
                )

    # Multi-suite / multi-task nested dict
    elif "per_group" in eval_info or "per_task" in eval_info:
        per_group = eval_info.get("per_group") or eval_info.get("per_task") or {}
        if isinstance(per_group, dict):
            for group_name, group_val in per_group.items():
                if isinstance(group_val, dict) and "pc_success" in group_val:
                    n_ep = int(group_val.get("n_episodes", 0) or 0)
                    pc = float(group_val.get("pc_success", 0))
                    tasks.append(
                        {
                            "task": str(group_name),
                            "task_id": None,
                            "suite": suite,
                            "n_episodes": n_ep,
                            "n_successes": int(round(pc / 100.0 * n_ep)) if n_ep else 0,
                            "pc_success": pc,
                            "seeds": [seed + i for i in range(n_ep)],
                            "video_paths": group_val.get("video_paths") or [],
                        }
                    )
                elif isinstance(group_val, dict):
                    for task_id, metrics in group_val.items():
                        if not isinstance(metrics, dict):
                            continue
                        n_ep = int(
                            metrics.get("n_episodes", 0) or len(metrics.get("per_episode", []))
                        )
                        pc = float(metrics.get("pc_success", 0))
                        n_ok = int(round(pc / 100.0 * n_ep)) if n_ep else 0
                        metric_episodes = metrics.get("per_episode", [])
                        seeds = [
                            int(ep.get("seed", seed + i))
                            for i, ep in enumerate(metric_episodes)
                            if isinstance(ep, dict)
                        ]
                        if not seeds:
                            seeds = [seed + i for i in range(n_ep)]
                        tasks.append(
                            {
                                "task": f"{group_name}/{task_id}",
                                "task_id": int(task_id) if str(task_id).isdigit() else None,
                                "suite": str(group_name),
                                "n_episodes": n_ep,
                                "n_successes": n_ok,
                                "pc_success": pc,
                                "seeds": seeds,
                                "video_paths": metrics.get("video_paths") or [],
                            }
                        )
                        for i, ep in enumerate(metrics.get("per_episode", [])):
                            if not isinstance(ep, dict):
                                continue
                            episodes.append(
                                {
                                    "episode_id": f"{group_name}_{task_id}_ep{ep.get('episode_ix', i)}",
                                    "task": f"{group_name}/{task_id}",
                                    "task_id": int(task_id) if str(task_id).isdigit() else None,
                                    "suite": str(group_name),
                                    "instruction": "",
                                    "seed": int(ep.get("seed", seed + i)),
                                    "success": bool(ep.get("success", False)),
                                    "policy_id": policy_id,
                                    "video_path": None,
                                    "sum_reward": ep.get("sum_reward"),
                                    "max_reward": ep.get("max_reward"),
                                }
                            )
    elif "aggregated" in eval_info or "per_episode" in eval_info:
        # Flat single-task format
        agg = eval_info.get("aggregated") or {}
        per_ep = eval_info.get("per_episode") or []
        n_ep = len(per_ep) or int(agg.get("n_episodes", 0) or 0)
        pc = float(agg.get("pc_success", 0))
        n_ok = (
            sum(1 for e in per_ep if e.get("success"))
            if per_ep
            else int(round(pc / 100.0 * n_ep))
        )
        video_paths = eval_info.get("video_paths") or []
        seeds = [
            int(ep.get("seed", seed + i))
            for i, ep in enumerate(per_ep)
            if isinstance(ep, dict)
        ]
        if not seeds:
            seeds = [seed + i for i in range(n_ep)]
        tasks.append(
            {
                "task": suite,
                "task_id": 0,
                "suite": suite,
                "n_episodes": n_ep,
                "n_successes": n_ok,
                "pc_success": pc,
                "seeds": seeds,
                "video_paths": video_paths,
            }
        )
        for i, ep in enumerate(per_ep):
            vid = video_paths[i] if i < len(video_paths) else None
            episodes.append(
                {
                    "episode_id": f"{suite}_ep{ep.get('episode_ix', i)}",
                    "task": suite,
                    "task_id": 0,
                    "suite": suite,
                    "instruction": "",
                    "seed": int(ep.get("seed", seed + i)),
                    "success": bool(ep.get("success", False)),
                    "policy_id": policy_id,
                    "video_path": vid,
                    "sum_reward": ep.get("sum_reward"),
                    "max_reward": ep.get("max_reward"),
                }
            )

    overall = None
    if "overall" in eval_info and isinstance(eval_info["overall"], dict):
        overall = float(eval_info["overall"].get("pc_success", 0))
    elif tasks:
        total_ep = sum(t["n_episodes"] for t in tasks)
        total_ok = sum(t["n_successes"] for t in tasks)
        overall = (100.0 * total_ok / total_ep) if total_ep else 0.0

    return {"tasks": tasks, "episodes": episodes, "overall_pc_success": overall}


@app.function(
    image=sim_image,
    gpu="L4",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: runs_volume},
)
def eval_checkpoint(
    policy_path: str,
    suite: str = "libero_object",
    task_ids: list[int] | None = None,
    n_episodes: int = 1,
    seed: int = 1000,
    run_id: str | None = None,
    n_action_steps: int = 10,
    batch_size: int = 1,
    max_parallel_tasks: int = 1,
    perturbation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run lerobot-eval inside the container; write results to the Volume."""
    import uuid
    from datetime import datetime, timezone

    run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    out_dir = Path(VOLUME_MOUNT) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Write HF token for huggingface_hub
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        Path.home().joinpath(".cache/huggingface").mkdir(parents=True, exist_ok=True)
        Path.home().joinpath(".cache/huggingface/token").write_text(hf_token)

    task_ids = task_ids if task_ids is not None else [0]
    task_ids_arg = "[" + ",".join(str(i) for i in task_ids) + "]"

    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids={task_ids_arg}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        f"--env.max_parallel_tasks={max_parallel_tasks}",
        f"--seed={seed}",
        f"--output_dir={out_dir}",
        f"--policy.n_action_steps={n_action_steps}",
    ]

    # Ensure LIBERO config exists even on cold volumes / fresh homes
    libero_cfg = Path.home() / ".libero" / "config.yaml"
    if not libero_cfg.exists():
        import importlib.util

        import yaml

        spec = importlib.util.find_spec("libero")
        if spec is None or not spec.origin:
            raise RuntimeError("libero package not installed in image")
        pkg = Path(spec.origin).resolve().parent
        root = pkg / "libero" if (pkg / "libero" / "bddl_files").exists() else pkg
        if not (root / "bddl_files").exists() and spec.submodule_search_locations:
            for p in [pkg, pkg / "libero", Path(list(spec.submodule_search_locations)[0]) / "libero"]:
                if (p / "bddl_files").exists():
                    root = p
                    break
        libero_cfg.parent.mkdir(parents=True, exist_ok=True)
        libero_cfg.write_text(
            yaml.dump(
                {
                    "benchmark_root": str(root),
                    "bddl_files": str(root / "bddl_files"),
                    "init_states": str(root / "init_files"),
                    "datasets": str(root.parent / "datasets"),
                    "assets": str(root / "assets"),
                }
            )
        )

    # Persist perturbation metadata for the suite (applied via env wrappers in v1)
    if perturbation:
        (out_dir / "perturbation.json").write_text(json.dumps(perturbation, indent=2))

    t0 = time.time()
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(out_dir),
    )
    gpu_seconds = time.time() - t0

    # Locate eval_info.json (lerobot may nest it)
    eval_info_path = None
    for candidate in [out_dir / "eval_info.json", *out_dir.rglob("eval_info.json")]:
        if candidate.is_file():
            eval_info_path = candidate
            break

    result: dict[str, Any] = {
        "run_id": run_id,
        "policy_id": policy_path,
        "suite": suite,
        "seed": seed,
        "n_episodes": n_episodes,
        "task_ids": task_ids,
        "gpu_seconds": gpu_seconds,
        "gpu_type": "L4",
        "status": "succeeded" if proc.returncode == 0 else "failed",
        "error": None if proc.returncode == 0 else (proc.stderr[-4000:] if proc.stderr else "eval failed"),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "eval_info": None,
        "tasks": [],
        "episodes": [],
        "overall_pc_success": None,
        "perturbation": perturbation,
    }

    if eval_info_path and eval_info_path.exists():
        eval_info = json.loads(eval_info_path.read_text())
        result["eval_info"] = eval_info
        parsed = _parse_eval_info(
            eval_info,
            suite=suite,
            policy_id=policy_path,
            seed=seed,
        )
        # Stamp perturbation onto tasks/episodes
        if perturbation:
            axis = perturbation.get("axis")
            level = int(perturbation.get("level", 0))
            for t in parsed["tasks"]:
                t["perturbation_axis"] = axis
                t["perturbation_level"] = level
            for e in parsed["episodes"]:
                e["perturbation_axis"] = axis
                e["perturbation_level"] = level
                e["perturbation_params"] = perturbation.get("params") or {}
                if perturbation.get("instruction"):
                    e["instruction"] = perturbation["instruction"]
        result.update(parsed)

    # Copy any videos into videos/ with unique names; stamp relative paths on result
    local_video_rels: list[str] = []
    for vid in out_dir.rglob("*.mp4"):
        if videos_dir in vid.parents or vid.parent == videos_dir:
            # Already under videos/
            rel = f"videos/{vid.relative_to(videos_dir).as_posix()}"
            if rel not in local_video_rels:
                local_video_rels.append(rel)
            continue
        # Prefer stable unique name: parent_dir + filename
        unique = f"{vid.parent.name}_{vid.name}" if vid.parent != out_dir else vid.name
        dest = videos_dir / unique
        if not dest.exists():
            dest.write_bytes(vid.read_bytes())
        rel = f"videos/{unique}"
        if rel not in local_video_rels:
            local_video_rels.append(rel)

    # Rewrite absolute /vol paths → relative videos/... for local pull/report
    if local_video_rels:
        for t in result.get("tasks") or []:
            t["video_paths"] = list(local_video_rels)
        eps = result.get("episodes") or []
        if eps:
            for i, e in enumerate(eps):
                e["video_path"] = local_video_rels[min(i, len(local_video_rels) - 1)]
        else:
            # Ensure episodes exist so reports can link videos even if parse missed them
            for i, rel in enumerate(local_video_rels):
                eps.append(
                    {
                        "episode_id": f"{suite}_ep{i}",
                        "task": suite,
                        "task_id": (task_ids or [0])[0],
                        "suite": suite,
                        "instruction": "",
                        "seed": seed + i,
                        "success": False,  # unknown; report shows all videos
                        "policy_id": policy_path,
                        "video_path": rel,
                    }
                )
            result["episodes"] = eps
        result["local_video_paths"] = local_video_rels

    (out_dir / "shard_result.json").write_text(json.dumps(result, indent=2, default=str))
    runs_volume.commit()
    return result


# Lightweight image for artifact pulls (no GPU / no lerobot)
_pull_image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=_pull_image,
    timeout=60 * 20,
    volumes={VOLUME_MOUNT: runs_volume},
)
def fetch_run_artifacts(
    run_id: str,
    *,
    max_videos: int = 40,
    max_bytes_per_video: int = 120 * 1024 * 1024,
) -> dict[str, Any]:
    """Read a run directory from the Volume and return file bytes for local materialization."""
    root = Path(VOLUME_MOUNT) / run_id
    if not root.exists():
        return {"ok": False, "error": f"run not found on volume: {run_id}", "files": {}, "videos": {}}

    text_files: dict[str, str] = {}
    videos: dict[str, bytes] = {}
    listed: list[str] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        listed.append(rel)
        if p.suffix.lower() == ".mp4":
            if len(videos) >= max_videos:
                continue
            data = p.read_bytes()
            if len(data) > max_bytes_per_video:
                continue
            videos[rel] = data
        elif p.suffix.lower() in {".json", ".txt", ".yaml", ".yml"} and p.stat().st_size < 5_000_000:
            try:
                text_files[rel] = p.read_text(errors="replace")
            except Exception:
                pass

    return {
        "ok": True,
        "run_id": run_id,
        "listed": listed,
        "files": text_files,
        "videos": videos,
        "n_videos": len(videos),
    }


@app.function(
    image=sim_image,
    gpu="L4",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: runs_volume},
)
def eval_shard(shard: dict[str, Any]) -> dict[str, Any]:
    """Thin wrapper so Modal .map() can fan out scenario shards."""
    return eval_checkpoint.local(
        policy_path=shard["policy_path"],
        suite=shard.get("suite", "libero_object"),
        task_ids=shard.get("task_ids"),
        n_episodes=shard.get("n_episodes", 1),
        seed=shard.get("seed", 1000),
        run_id=shard.get("run_id"),
        n_action_steps=shard.get("n_action_steps", 10),
        batch_size=shard.get("batch_size", 1),
        perturbation=shard.get("perturbation"),
    )


@app.function(
    image=sim_image,
    timeout=60 * 30,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: runs_volume},
)
def derisk_one_episode(
    policy_path: str = "HuggingFaceVLA/smolvla_libero",
    suite: str = "libero_object",
) -> dict[str, Any]:
    """Day-1 derisk: one episode, dump eval_info schema."""
    return eval_checkpoint.local(
        policy_path=policy_path,
        suite=suite,
        task_ids=[0],
        n_episodes=1,
        seed=1000,
        run_id="derisk_day1",
        n_action_steps=10,
    )


@app.local_entrypoint()
def main(
    policy: str = "HuggingFaceVLA/smolvla_libero",
    suite: str = "libero_object",
    episodes: int = 1,
    seed: int = 1000,
    task_ids: str = "0",
    derisk: bool = False,
    pull: bool = True,
):
    ids = [int(x) for x in task_ids.split(",") if x.strip() != ""]
    run_id = "derisk_day1" if derisk else None
    # Always call eval_checkpoint.remote directly (avoid nested .local() indirection)
    result = eval_checkpoint.remote(
        policy_path=policy,
        suite=suite,
        task_ids=ids if not derisk else [0],
        n_episodes=1 if derisk else episodes,
        seed=seed,
        run_id=run_id,
        n_action_steps=10,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("eval_info",)}, indent=2, default=str))
    if result.get("eval_info"):
        print("--- eval_info keys ---")
        print(list(result["eval_info"].keys()))

    if pull and result.get("run_id"):
        # Materialize videos + JSON onto local runs/
        from policyci.runner.client import materialize_run_from_result

        # Pull after remote returns — volume get, not nested RPC
        local_dir = materialize_run_from_result(
            result, pull_videos=True, use_existing_app=False
        )
        print(f"--- pulled to {local_dir} ---")
        vids = list((local_dir / "videos").rglob("*.mp4")) if (local_dir / "videos").exists() else []
        print(f"local videos: {len(vids)}")
        for v in vids[:10]:
            print(f"  {v}")
    return result
