from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


try:
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None


@dataclass
class DoorController:
    """Controls a door via Arduino over serial, or simulates if unavailable."""

    simulate: bool
    port: str
    baudrate: int
    _ser: Optional[object] = None

    def connect(self) -> None:
        """Connect to Arduino if simulation is disabled."""
        if self.simulate:
            self._ser = None
            return

        if serial is None:
            raise RuntimeError("pyserial not installed; enable simulation or install it.")

        self._ser = serial.Serial(self.port, self.baudrate, timeout=1)

    def _write(self, command: str) -> None:
        """Write a command to Arduino or print in simulation mode."""
        command = command.strip().upper()

        if self.simulate or self._ser is None:
            if command == "OPEN":
                print("Door Opened (SIMULATION)")
            elif command == "CLOSE":
                print("Door Closed (SIMULATION)")
            else:
                print(f"Arduino Command (SIMULATION): {command}")
            return

        self._ser.write((command + "\n").encode("utf-8"))

    def open_door(self) -> None:
        """Open door (rotate servo)."""
        self._write("OPEN")

    def close_door(self) -> None:
        """Close door (return servo)."""
        self._write("CLOSE")

    def disconnect(self) -> None:
        """Close the serial connection if open."""
        if self._ser is not None and hasattr(self._ser, "close"):
            try:
                self._ser.close()
            finally:
                self._ser = None

