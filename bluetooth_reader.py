# =============================================================================
# bluetooth_reader.py — OptoScan Bluetooth / Serial Reader
# =============================================================================
# Runs a background daemon thread that continuously reads integer ADC values
# from the HC-05 Bluetooth module (paired as a Virtual COM port).
#
# Features:
#   • Non-blocking: uses queue.Queue so the UI thread is never stalled
#   • Exponential back-off reconnect on port loss (1s → 2s → 4s … capped 30s)
#   • Counts data drops (invalid frames or read timeouts)
#   • Thread-safe connection status via threading.Event
# =============================================================================

import threading
import queue
import time
import logging

try:
    import serial
    import serial.serialutil
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from config import (
    BAUD_RATE, READ_TIMEOUT,
    RECONNECT_BASE, RECONNECT_MAX,
    QUEUE_MAX_SIZE, ADC_MIN, ADC_MAX,
)

logger = logging.getLogger(__name__)


class BluetoothReader:
    """
    Reads raw ADC integers from an Arduino / HC-05 over a Virtual COM port.

    Usage
    -----
    reader = BluetoothReader(port="COM3")
    reader.start()

    while True:
        value = reader.latest_value()   # None if no data yet
        ...

    reader.stop()
    """

    def __init__(self, port: str, baud: int = BAUD_RATE):
        self.port = port
        self.baud = baud

        self._data_queue: "queue.Queue[int]" = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self._stop_event  = threading.Event()
        self._connected   = threading.Event()
        self._thread: threading.Thread | None = None

        # Diagnostics
        self._total_samples  = 0
        self._drop_count     = 0
        self._last_raw: int | None = None
        self._last_timestamp: float | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background reading thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("BluetoothReader already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="BT-Reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("BluetoothReader started on %s @ %d baud", self.port, self.baud)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._connected.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("BluetoothReader stopped.")

    def is_connected(self) -> bool:
        """True while the COM port is open and receiving data."""
        return self._connected.is_set()

    def latest_value(self) -> "int | None":
        """
        Drain the queue and return the most recent valid ADC sample,
        or None if no data is available.
        """
        value = None
        try:
            while True:
                value = self._data_queue.get_nowait()
        except queue.Empty:
            pass
        return value

    def get_queue(self) -> "queue.Queue[int]":
        """Direct access to the raw queue (for advanced consumers)."""
        return self._data_queue

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def drop_count(self) -> int:
        return self._drop_count

    @property
    def last_timestamp(self) -> "float | None":
        return self._last_timestamp

    # ── Internal ───────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main loop: connect → read → reconnect on error."""
        backoff = RECONNECT_BASE

        while not self._stop_event.is_set():
            ser = None
            try:
                ser = self._open_port()
                if ser is None:
                    # Serial library not available — run in simulation mode
                    self._simulate(backoff)
                    backoff = min(backoff * 2, RECONNECT_MAX)
                    continue

                self._connected.set()
                backoff = RECONNECT_BASE          # reset back-off on success
                logger.info("Connected to %s", self.port)

                self._read_loop(ser)

            except serial.SerialException as exc:
                logger.warning("Serial error on %s: %s", self.port, exc)
            except Exception as exc:
                logger.error("Unexpected error in reader thread: %s", exc)
            finally:
                self._connected.clear()
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass

            if not self._stop_event.is_set():
                logger.info("Reconnecting in %.1fs …", backoff)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, RECONNECT_MAX)

    def _open_port(self):
        """Attempt to open the serial port. Returns None if pyserial missing."""
        if not SERIAL_AVAILABLE:
            return None
        return serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=READ_TIMEOUT,
        )

    def _read_loop(self, ser) -> None:
        """Inner loop that reads lines from an open serial port."""
        while not self._stop_event.is_set():
            try:
                raw_line = ser.readline()
                if not raw_line:
                    # Timeout — no data received within READ_TIMEOUT window
                    self._drop_count += 1
                    continue

                text = raw_line.decode("ascii", errors="ignore").strip()
                value = int(text)

                # Clamp to valid ADC range
                if not (ADC_MIN <= value <= ADC_MAX):
                    logger.debug("Out-of-range sample discarded: %d", value)
                    self._drop_count += 1
                    continue

                self._last_raw = value
                self._last_timestamp = time.time()
                self._total_samples += 1

                # Non-blocking put — drop oldest if queue is full
                if self._data_queue.full():
                    try:
                        self._data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._drop_count += 1

                self._data_queue.put_nowait(value)

            except ValueError:
                # Malformed frame (partial line, non-integer)
                self._drop_count += 1
            except serial.SerialException:
                # Port pulled — break inner loop to trigger reconnect
                raise

    def _simulate(self, wait: float) -> None:
        """
        Stand-in generator when pyserial is not installed.
        Produces a slow sine-wave for UI testing.
        """
        import math
        import random
        logger.info("pyserial not available — running in SIMULATION mode")
        t = 0
        while not self._stop_event.is_set() and t < 50:
            # Gentle oscillation across the full ADC range + small noise
            raw = int(512 + 400 * math.sin(t * 0.2) + random.gauss(0, 10))
            raw = max(ADC_MIN, min(ADC_MAX, raw))
            self._data_queue.put_nowait(raw)
            self._total_samples += 1
            self._last_timestamp = time.time()
            time.sleep(0.1)
            t += 1
