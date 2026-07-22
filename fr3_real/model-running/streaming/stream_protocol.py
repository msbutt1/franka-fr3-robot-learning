"""Versioned UDP protocol shared by FR3 streaming tools.
# Usage: import this shared module from an FR3 script; it is not a standalone command.

Plans are tied to the monotonic timestamp of the robot state used to produce
them. The streamer uses that timestamp to skip actions that became stale while
the policy was running.
"""

from __future__ import annotations

import dataclasses
import socket
import struct
from typing import Final

import numpy as np


COMMAND_MAGIC: Final = 0x46523343  # FR3C
STATE_MAGIC: Final = 0x46523353  # FR3S
VERSION: Final = 3
MAX_STEPS: Final = 16

COMMAND_PACKET: Final = struct.Struct("<IIQQBBH" + "d" * (MAX_STEPS * 7))
STATE_PACKET: Final = struct.Struct("<IIQ" + "d" * 17 + "QIIII")

STOP_REASONS: Final = {
    0: "none",
    1: "command",
    2: "signal",
    3: "joint_guard",
    4: "workspace_guard",
    5: "watchdog",
    6: "predicted_guard",
    7: "stale_plan",
}


@dataclasses.dataclass(frozen=True)
class StreamState:
    monotonic_ns: int
    q: np.ndarray
    dq: np.ndarray
    tcp: np.ndarray
    active_sequence: int
    plan_step: int
    plan_steps: int
    activation_offset: int
    stop_reason: int

    @property
    def stop_reason_name(self) -> str:
        return STOP_REASONS.get(self.stop_reason, f"unknown_{self.stop_reason}")


def make_command(
    sequence: int,
    observation_monotonic_ns: int,
    velocities: np.ndarray | None,
) -> bytes:
    values = np.zeros((MAX_STEPS, 7), dtype=np.float64)
    stop = velocities is None
    steps = 0
    if velocities is not None:
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.ndim != 2 or velocities.shape[1] != 7:
            raise ValueError(f"velocities must have shape (steps, 7), got {velocities.shape}")
        steps = len(velocities)
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError("stream plan must contain 1..16 actions")
        if not np.isfinite(velocities).all():
            raise ValueError("stream plan contains NaN or infinite actions")
        values[:steps] = velocities
    return COMMAND_PACKET.pack(
        COMMAND_MAGIC,
        VERSION,
        int(sequence),
        int(observation_monotonic_ns),
        int(stop),
        steps,
        0,
        *values.ravel(),
    )


def unpack_state(payload: bytes) -> StreamState | None:
    if len(payload) != STATE_PACKET.size:
        return None
    fields = STATE_PACKET.unpack(payload)
    if fields[0] != STATE_MAGIC or fields[1] != VERSION:
        return None
    return StreamState(
        monotonic_ns=int(fields[2]),
        q=np.asarray(fields[3:10], dtype=np.float64),
        dq=np.asarray(fields[10:17], dtype=np.float64),
        tcp=np.asarray(fields[17:20], dtype=np.float64),
        active_sequence=int(fields[20]),
        plan_step=int(fields[21]),
        plan_steps=int(fields[22]),
        activation_offset=int(fields[23]),
        stop_reason=int(fields[24]),
    )


def receive_latest_state(sock: socket.socket) -> tuple[StreamState, list[StreamState]]:
    """Receive one state, then drain the UDP backlog and return the newest.

    The complete drained list is returned so callers can persist every actual
    state packet while still making decisions from only the freshest state.
    """

    received: list[StreamState] = []
    while not received:
        payload, _ = sock.recvfrom(512)
        state = unpack_state(payload)
        if state is not None:
            received.append(state)
    configured_timeout = sock.gettimeout()
    sock.setblocking(False)
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(512)
            except BlockingIOError:
                break
            state = unpack_state(payload)
            if state is not None:
                received.append(state)
    finally:
        sock.settimeout(configured_timeout)
    return received[-1], received
