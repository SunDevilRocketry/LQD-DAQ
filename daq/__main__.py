"""
daq/__main__.py

Run with:
    python -m daq
    python -m daq --config path/to/custom_config.yaml
    python -m daq --mock          (force mock hardware regardless of LJM)
    python -m daq --no-server     (engine only, no HTTP API)

Startup sequence:
    1. Parse CLI args and load config.yaml
    2. Load the channel/actuator manifests
    3. Instantiate Engine (owns the manifests and the device)
    4. Instantiate Logger against the engine's channel manifest
    5. Register API engine reference
    6. Start engine (connects to hardware, starts stream thread)
    7. Start uvicorn (serves API; blocks until Ctrl-C)
    8. On shutdown: stop engine, flush logger.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import yaml
import uvicorn

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
    },
    "stream":  {"target_hz": 500},
    "thresholds": {},  # Allows the parser to recognize and merge this section
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
    """Make a path absolute relative to the working directory."""
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
    if args.mock:
        # Override the hardware classes before other modules import them
        import daq.hardware as _hw
        from daq.hardware.mock import MockLabJack
        _hw.Device     = MockLabJack
        _hw.USING_MOCK = True
        print("[MAIN] Forced mock hardware")

    # Now safe to import Engine and API after hardware references are resolved
    from daq.engine import Engine
    from daq import api as api_module

    # -- 3. Initialize Engine (loads the cart manifests) -----
    cal_path     = _resolve(paths["cal_file"])
    sequence_dir = _resolve(paths["sequence_dir"])

    engine = Engine(
        cal_path=cal_path,
        sequence_dir=sequence_dir,
        thresholds=cfg.get("thresholds"),  # Ingest thresholds from custom_config.yaml / config.yaml
        channels_path=_resolve(paths["channels"]),
        actuators_path=_resolve(paths["actuators"]),
    )
    print(f"[MAIN] Manifest: {len(engine.channel_specs)} channels, "
          f"{len(engine.actuator_specs)} actuators")

    # -- 4. Initialize CSV Logger ----------------------------
    # CSV schema follows the engine's channel manifest.
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
            sys.exit(1)         # Force termination on consecutive signals
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
