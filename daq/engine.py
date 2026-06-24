"""
daq/engine.py
 
500 Hz background acquisition engine for the Liquids DAQ system.
 
Responsibilities:
  - Drives the hardware stream loop in a dedicated daemon thread
  - Applies calibration and calculations to every scan in real time
  - Maintains a thread-safe snapshot of the latest engineering values
  - Runs fire / abort autosequences on a separate high-priority thread
  - Triggers and stops the CSV logger (daq/logger.py)
  - Exposes a simple read-only API for api.py and the GUI to consume
 
Design constraints (from SR 3.4.1):
  - Logging is fully decoupled from the display rate
  - All 500 Hz scans are logged; the API/GUI may poll at any lower rate
  - Sequence timing uses perf_counter, not sleep, to avoid drift
 
Threading model:
  - _stream_thread    — calls stream_read() in a tight loop, processes batches
  - _sequence_thread  — runs autosequences; writes actuators via device
  - Main / API thread — reads self.snapshot (always safe, never blocks)
 
All shared mutable state is protected by self._lock (a RLock so the
sequence thread can call write_actuator which also acquires it).
"""

# Should in theory do that ^^^^

# ------------------------------------------------------------------------------
# Multi-Threading Model Design
# ------------------------------------------------------------------------------
# - _stream_thread (daemon):
#   * Tight loop calling T7 stream_read().
#   * Applies calibration math on every raw scan.
#   * Packages results into an atomic snapshot (EngineState) for lock-free UI polling.
#   * Periodically reads CJC (LM34) outside the stream (0.5s interval to minimize lag).
#
# - _sequence_thread (high priority):
#   * Runs fire / abort autosequences.
#   * Uses time.perf_counter() polling to target ~1ms precision (avoid time.sleep drift).
#   * Actuator writes are owned by this thread during sequence; manual overrides blocked.
#
# - Main / API thread: 
#   * Reads atomically replaced EngineState snapshots. Never blocks.

# ------------------------------------------------------------------------------
# Calculation Pipeline
# ------------------------------------------------------------------------------
# For each raw scan in the stream buffer batch:
#   * PT Conversion: psi = volts * slope + intercept (dynamic scaling per sensor).
#   * TC Conversion: Software Seebeck Type-K calculation using CJC as reference.
#     - Note: TCs are high-frequency noisy. Average TC voltages across the entire
#       scan batch (e.g., 50 scans) before applying Seebeck math to prevent noise spikes.
#   * LC Conversion: force = volts * slope + intercept - tare.
#   * Impulse Integration:
#     - Integrate area under total load cell force curve per scan (trapezoidal step).
#     - Fallback: If load cells are disconnected, calculate estimated impulse step
#       using calculated lox/fuel mass flow rates and expected Isp.
#   * Derived Values:
#     - lox_mdot = function(TOI, POI, PC)
#     - fuel_mdot = function(PFO, PC)
#     - mixture_ratio = lox_mdot / fuel_mdot
#     - Saturation: Check if LOX is subcooled based on pressure/temperature (lox_below_sat).

# ------------------------------------------------------------------------------
# YAML Interface :) YT tutorial looked fire
# ------------------------------------------------------------------------------
# - Sequence files (fire.yaml / abort.yaml) will dictate testing timing.
# - Structure:
#   * post_record_seconds: record buffer data for N seconds after sequence completes.
#   * steps: sorted array of [elapsed_seconds, actuator_name, target_state].
# - Thread loop must poll elapsed time at ~1ms interval
# - Safety Interlock: Abort flag must instantly kill the fire loop and safe the stand.

# ------------------------------------------------------------------------------
# Logger & Calibration Decoupling
# ------------------------------------------------------------------------------
# - Read default calibration values from config JSON on startup; merge dynamic changes.
# - Allow dynamic updates to slope/intercept without stopping the stream.
# - Decouple CSV file I/O from stream thread to prevent disk write latencies from
#   overflowing the T7 stream buffer.

# ------------------------------------------------------------------------------
# TODO
# ------------------------------------------------------------------------------
# thread-safe RLock layout for manual actuator command overrides.
# math modules for Type-K conversion and fluid densities.
# Test timer resolution and latency on target Linux test stand PC.
# Implement circular queue to buffer debug messages for console output.