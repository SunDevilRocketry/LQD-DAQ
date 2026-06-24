"""
daq/hardware/interface.py
 
Production LabJack T7 driver wrapping the LJM library.
 
Should match hardware/mock.py interface to maintain engine.py agnostic.
 
Channel layout (must match mock.py and engine.py):
    PTs  — single-ended, AIN_NEGATIVE_CH=199, ±5 V range
    TCs  — differential pairs, ±0.1 V range
    LCs  — single-ended, AIN_NEGATIVE_CH=199, ±5 V range
    CJC  — LM34 on AIN58, single-ended, ±1 V range (read outside stream)
 
Actuators (active-low on EIO/CIO bank):
    hardware logical 0 = energised, logical 1 = de-energised
    We invert at this layer so the rest of the system uses:
        1 = open/energised, 0 = closed/safe
 
Watchdog:
    Armed during start_stream(). If the host loses comms for > 5 s the
    T7 drives all DIO lines to their safe (de-energised) defaults.
 
Reference:
    LabJack T7 User Guide — https://labjack.com/pages/support?doc=/datasheets/t7-datasheet/
    LJM Library — https://labjack.com/pages/support?doc=/software-driver/ljm-users-guide/
"""

# ------------------------------------------------------------------------------
# Physical Channel Allocations 
# ------------------------------------------------------------------------------
# PTs (Pressure Transducers): Single-ended, GND reference (NEG = 199), +/- 5V range.
#   - POT (LOX Tank)         -> AIN122
#   - PFT (Fuel Tank)        -> AIN123
#   - POI (LOX Inlet)        -> AIN124
#   - PFI (Fuel Inlet)       -> AIN125
#   - PFO (Fuel Outlet)      -> AIN126
#   - PC  (Chamber)          -> AIN127
#   - PNS (Sys GN2)          -> AIN52
#   - PNP (Pneumatics GN2)   -> AIN53
#
# TCs (Thermocouples): Differential pairs, +/- 0.1V range for small signals.
#   - TOI (LOX Inlet)        -> AIN48 / AIN56 (Positive / Negative pair)
#   - TFI (Fuel Inlet)       -> AIN0  / AIN1
#   - TFO (Fuel Outlet)      -> AIN2  / AIN3
#
# LCs (Load Cells): Single-ended on AIN50, AIN51.
#
# CJC (Cold Junction): LM34 on AIN58, +/- 1V range. Single-ended.
#   * Note: CJC should NOT be in the main stream list. Read it outside the stream 
#     loop via eReadName to avoid Mux80 switching delays.

# ------------------------------------------------------------------------------
# Actuator Stuff
# ------------------------------------------------------------------------------
# - Valves: LOx/Fuel Press, Purge, Main, Vent mapped to EIO0 - EIO7.
# - Igniter: Mapped to CIO1.
# - NOTE: Hardware is ACTIVE-LOW. 
#   * 0 = Energized/Open, 1 = De-energized/Safe.
#   * The driver must invert this logical state internally. 
#   * Callers should use 1 for open/energized, 0 for closed/safe.

# ------------------------------------------------------------------------------
# Timing parameter *double check*
# ------------------------------------------------------------------------------
# - Target Rate: 500 Hz
# - Scans per read: 50 (aiming for ~10 batches/sec, or 0.1s latency per read call).
# - Settling time: 100 microseconds.
# - Resolution index: 3 (higher resolution, lower noise—verify if 500Hz can handle).

# ------------------------------------------------------------------------------
#  Hardware Watchdog [sounds cool] (Safety Interlock)
# ------------------------------------------------------------------------------
# - Watchdog timeout: 5 seconds. If host stops comms, T7 must go safe autonomously.
# - Register configuration sequence:
#   1. WATCHDOG_ENABLE_DEFAULT = 0 (disable while configuring)
#   2. WATCHDOG_TIMEOUT_S_DEFAULT = 5
#   3. WATCHDOG_DIO_ENABLE_DEFAULT = 1
#   4. WATCHDOG_DIO_INHIBIT_DEFAULT: Bitmask to isolate actuator pins. 
#      EIO0-7 (bits 8-15) + CIO1 (bit 17). Mask = 0x0002FF00. Invert for inhibit.
#   5. WATCHDOG_DIO_DIRECTION_DEFAULT: Set actuator pins to outputs.
#   6. WATCHDOG_DIO_STATE_DEFAULT: Set default safe state (all 1s for active-low safe).
#   7. WATCHDOG_ENABLE_DEFAULT = 1 (arm the watchdog)

# ------------------------------------------------------------------------------
# Prob do following:
# ------------------------------------------------------------------------------
# = LJM SDK wrapper structure and mock verification.
# = Write stream fallback logic: if target HZ fails with SCAN_OVERLAP (code 2942), 
#     halve the target rate and re-attempt.
# = clean exit method to ensure all actuators are driven high (safe) on close.
# = Verify differential TC offset calculations using the CJC temperature logic.
# = Test physical watchdog timeout behavior with dummy LED load.