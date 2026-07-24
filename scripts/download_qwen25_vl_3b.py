#!/usr/bin/env python3
"""Resumable background download for the locked EXP-001 development model."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from modelscope import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "master"
RETRY_SECONDS = 15


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    attempt = 0
    print(
        f"[{timestamp()}] downloader started PID={os.getpid()} "
        f"model={MODEL_ID} local_dir={LOCAL_DIR}",
        flush=True,
    )

    while True:
        attempt += 1
        try:
            print(f"[{timestamp()}] attempt={attempt}", flush=True)
            result = snapshot_download(
                model_id=MODEL_ID,
                revision=REVISION,
                local_dir=str(LOCAL_DIR),
                max_workers=8,
            )
            print(f"[{timestamp()}] COMPLETE local_dir={result}", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(
                f"[{timestamp()}] RETRY_AFTER_ERROR "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(RETRY_SECONDS)


if __name__ == "__main__":
    main()
