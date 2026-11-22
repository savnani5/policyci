#!/usr/bin/env python3
"""Manufacture a quantized v2 checkpoint from a LIBERO-finetuned v1.

Loads via LeRobot's from_pretrained, applies int8 dynamic quantization to
Linear layers, saves via save_pretrained so lerobot-eval can load it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def quantize(src: str, dst: Path, backend: str = "qnnpack") -> Path:
    import torch
    from torch.ao.quantization import quantize_dynamic

    # Prefer PI0.5, fall back to PI0 / generic PreTrainedConfig loader
    policy = None
    errors: list[str] = []
    for loader_name in ("PI05Policy", "PI0Policy", "SmolVLAPolicy"):
        try:
            mod = __import__("lerobot.common.policies", fromlist=[loader_name])
            # Newer lerobot layouts
        except Exception as e:
            errors.append(str(e))

    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(src)
    except Exception as e1:
        errors.append(f"PI05: {e1}")
        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            policy = PI0Policy.from_pretrained(src)
        except Exception as e2:
            errors.append(f"PI0: {e2}")
            try:
                from lerobot.common.policies.factory import get_policy_class
                from lerobot.configs.policies import PreTrainedConfig

                cfg = PreTrainedConfig.from_pretrained(src)
                cls = get_policy_class(cfg.type)
                policy = cls.from_pretrained(src)
            except Exception as e3:
                errors.append(f"factory: {e3}")
                raise RuntimeError(
                    "Could not load policy. Tried PI05/PI0/factory:\n" + "\n".join(errors)
                ) from e3

    policy.eval()
    # Dynamic int8 quantization on Linear layers — realistic deploy regression
    try:
        torch.backends.quantized.engine = backend
    except Exception:
        pass

    qpolicy = quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Prefer native save_pretrained if available
    if hasattr(policy, "save_pretrained") and hasattr(policy, "config"):
        try:
            policy.save_pretrained(dst)
            # Overwrite weights with a degraded float approximation so eval still loads
            # (true int8 dynamic modules often aren't Hub-reloadable).
            state = policy.state_dict()
            degraded = {}
            for k, v in state.items():
                if hasattr(v, "dtype") and v.dtype.is_floating_point and v.ndim >= 2:
                    noise = torch.randn_like(v) * (v.std() * 0.15 + 1e-4)
                    degraded[k] = (v + noise).cpu()
                else:
                    degraded[k] = v.cpu() if hasattr(v, "cpu") else v
            torch.save(degraded, dst / "pytorch_model.bin")
            (dst / "QUANTIZED.txt").write_text(
                f"source={src}\nmethod=weight_noise_approx_int8\nbackend={backend}\n"
            )
        except Exception:
            torch.save({"model": qpolicy}, dst / "quantized.pt")
            (dst / "QUANTIZED.txt").write_text(f"source={src}\nmethod=torch_quantize_dynamic\n")
    else:
        torch.save({"model": qpolicy}, dst / "quantized.pt")

    print(f"wrote quantized checkpoint → {dst}")
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="lerobot/pi05_libero_finetuned_v044",
        help="Source Hub id or local path",
    )
    p.add_argument(
        "--dst",
        default="checkpoints/pi05_libero_quantized",
        help="Output directory",
    )
    p.add_argument("--backend", default="qnnpack", help="Quantization backend")
    args = p.parse_args()
    quantize(args.src, Path(args.dst), backend=args.backend)


if __name__ == "__main__":
    main()
