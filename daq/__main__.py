"""
daq/__main__.py

    python -m daq [--config FILE] [--mock] [--no-server]

Without LJM the service refuses to start unless --mock is given.
Startup: config -> Engine (manifests, device) -> Logger -> API -> engine
start -> uvicorn. Shutdown stops the engine, then the logger.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import yaml
import uvicorn

from daq.calculations import load_lox_table
from daq.logger import Logger


# -- Defaults (overridden by config.yaml) --------------------

_DEFAULT_CONFIG = {
    "server":  {"host": "0.0.0.0", "port": 8000, "log_level": "warning"},
    "paths":   {
        "output_dir":   "data",
        "sequence_dir": "sequences",
        "cal_file":     "daq/calibration.json",
        "channels":     "channels.yaml",
        "actuators":    "actuators.yaml",
        "lox_table":    "daq/LOX_table_100_0-260_99R.csv",
    },
    "stream":  {"target_hz": 500},
    "thresholds": {},  # listed so config.yaml merges into them
    "derived": {},
}


def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[MAIN] Config not found at '{path}', using defaults")
        return _DEFAULT_CONFIG
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    
    cfg = dict(_DEFAULT_CONFIG)
    for section, content in user.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(content, dict):
            cfg[section] = {**cfg[section], **content}
        else:
            cfg[section] = content
            
    return cfg


def _resolve(path: str) -> str:
    """Absolute path from the working directory."""
    return os.path.abspath(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m daq",
        description="Liquids DAQ - ground controller acquisition service",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Force mock hardware (useful for UI development on a bench laptop)"
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="Run engine only without starting the HTTP API server"
    )
    args = parser.parse_args()

    # -- 1. Load Configurations ------------------------------
    cfg     = _load_config(args.config)
    paths   = cfg["paths"]
    server  = cfg["server"]

    # -- 2. Hardware Mock Configuration ----------------------
    # daq.hardware falls back to the mock without LJM; refuse that unless asked.
    import daq.hardware as _hw
    if _hw.USING_MOCK and not args.mock:
        print("[MAIN] ERROR: LabJack LJM native library not found, so only "
              "mock hardware is available. Install LJM to use the T7, or "
              "pass --mock to run on simulated data.")
        sys.exit(1)

    if args.mock:
        # Before daq.engine imports Device.
        from daq.hardware.mock import MockLabJack
        _hw.Device     = MockLabJack
        _hw.USING_MOCK = True
        print("[MAIN] Forced mock hardware")

    from daq.engine import Engine
    from daq import api as api_module

    # -- 3. Initialize Engine (loads the cart manifests) -----
    cal_path     = _resolve(paths["cal_file"])
    sequence_dir = _resolve(paths["sequence_dir"])

    # Without it, LOX mass flow is null.
    lox_table_path = _resolve(paths["lox_table"])
    if os.path.exists(lox_table_path):
        try:
            n, lo, hi = load_lox_table(lox_table_path)
            print(f"[MAIN] LOX table loaded: {n} points, {lo:.1f}–{hi:.1f} R")
        except Exception as exc:
            print(f"[MAIN] WARNING: LOX table load failed: {exc}")
    else:
        print(f"[MAIN] WARNING: LOX table not found at {lox_table_path}")

    engine = Engine(
        cal_path=cal_path,
        sequence_dir=sequence_dir,
        thresholds=cfg.get("thresholds"),
        derived=cfg.get("derived"),
        channels_path=_resolve(paths["channels"]),
        actuators_path=_resolve(paths["actuators"]),
        stream_hz=int(cfg["stream"]["target_hz"]),
        # Mock only: placeholder step counts so sequences run without hardware.
        mock_stepper_defaults=_hw.USING_MOCK,
    )
    print(f"[MAIN] Manifest: {len(engine.channel_specs)} channels, "
          f"{len(engine.actuator_specs)} actuators")

    # -- 4. Initialize CSV Logger ----------------------------
    output_dir = _resolve(paths["output_dir"])
    logger = Logger(output_dir=output_dir, channels=engine.channel_specs)
    logger.open()
    engine.set_logger(logger)
    print(f"[MAIN] Logger ready -> {output_dir}")

    # -- 5. Bind API References ------------------------------
    api_module.set_engine(engine, logger)

    # -- 6. Signal Interruption Handlers ---------------------
    _shutdown_requested = [False]

    def _shutdown(sig, frame):
        if _shutdown_requested[0]:
            sys.exit(1)         # second signal: exit now
        _shutdown_requested[0] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # -- 7. Launch Engine Acquisition ------------------------
    engine.start()
    print(f"[MAIN] Engine running (mock={api_module._engine.snapshot.using_mock})")

    # -- 8. Run HTTP API Server or Headless Loop -------------
    try:
        if args.no_server:
            print("[MAIN] Running headless (no HTTP server). Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
                snap = engine.snapshot
                if snap.streaming:
                    readings = " ".join(
                        f"{cid}="
                        + (f"{r.value:.1f}{r.unit}" if r.value is not None else "-")
                        for cid, r in (snap.channels or {}).items()
                    )
                    print(f"  {readings}  seq={snap.sequence_active}")
        else:
            print(f"[MAIN] API server at http://{server['host']}:{server['port']}")
            print("[MAIN] Press Ctrl+C to stop.")
            uvicorn.run(
                "daq.api:app",
                host=server["host"],
                port=server["port"],
                log_level=server["log_level"],
            )
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted - exiting…")
    finally:
        print("[MAIN] Stopping engine and flushing logger…")
        engine.stop()
        logger.close()
        print("[MAIN] Clean shutdown complete.")

if __name__ == "__main__":
    main()
