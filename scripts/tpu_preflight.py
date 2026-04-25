#!/usr/bin/env python3
# Usage: python scripts/tpu_preflight.py
import os
import shutil
import sys
ok = 0
try:
    import jax
    devices = jax.devices()
    device_lines = [f"{i}: {getattr(d, 'device_kind', '?')} {d}" for i, d in enumerate(devices)]
    if len(devices) != 8:
        raise RuntimeError(f"expected 8 devices, found {len(devices)}")
    bad = [getattr(d, "device_kind", "") for d in devices if "TPU v6" not in getattr(d, "device_kind", "")]
    if bad:
        raise RuntimeError(f"unexpected device_kind values: {bad}")
    print("[OK] jax.devices(): " + "; ".join(device_lines)); ok += 1
except Exception as exc:
    print(f"[FAIL: {exc}] jax.devices()")
try:
    import flax
    import flax.nnx as nnx
    print(f"[OK] flax import: flax {flax.__version__}, nnx {nnx.__name__}"); ok += 1
except Exception as exc:
    print(f"[FAIL: {exc}] flax import")
try:
    import easydel
    easydel_path = getattr(easydel, "__file__", "") or ""
    if "easydel" not in easydel_path.lower():
        raise RuntimeError(f"editable install path check failed: {easydel_path}")
    print(f"[OK] easydel import: {easydel_path}"); ok += 1
except Exception as exc:
    print(f"[FAIL: {exc}] easydel import")
try:
    from huggingface_hub import whoami
    identity = whoami()
    username = identity.get("name") or identity.get("fullname") or identity.get("email")
    if not username:
        raise RuntimeError(f"authenticated but username missing: {identity}")
    print(f"[OK] huggingface auth: {username}"); ok += 1
except Exception as exc:
    print(f"[FAIL: unauthenticated or token invalid: {exc}] huggingface auth")
try:
    paths = ["/"]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        paths.append(hf_home)
    details = []
    for path in paths:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        details.append(f"{path}: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        if free_gb < 30:
            raise RuntimeError(f"{path} has {free_gb:.1f} GB free, need >= 30.0 GB")
    print("[OK] disk usage: " + "; ".join(details)); ok += 1
except Exception as exc:
    print(f"[FAIL: {exc}] disk usage")
try:
    import psutil
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    print(f"[OK] memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available"); ok += 1
except Exception as exc:
    print(f"[FAIL: {exc}] memory")
try:
    jax.print_environment_info()
    print("[OK] jax.print_environment_info()"); ok += 1
except Exception as exc:
    print(f"[FAIL: {exc}] jax.print_environment_info()")
status = "OK" if ok == 7 else "FAIL"
print(f"PREFLIGHT: {ok}/7 {status}")
sys.exit(0 if ok == 7 else 1)
