"""Per-frame state assembly and JSON/Parquet export."""

import json
from pathlib import Path

import numpy as np


def build_frame_state(
    frame_id: int,
    timestamp_s: float,
    players: list[dict],
    balls: list[dict],
    court_mapper=None,
) -> dict:
    """
    Assemble a structured state dict for one frame.

    Each player dict is expected to have:
        bbox, conf, track_id, team, pose (optional)
    """
    player_states = []
    for p in players:
        bbox = p["bbox"]
        foot_x = (bbox[0] + bbox[2]) / 2.0
        foot_y = float(bbox[3])

        court_pos = None
        on_court = True
        if court_mapper and court_mapper.is_calibrated:
            court_pos = court_mapper.pixel_to_court(foot_x, foot_y)
            on_court = court_mapper.is_on_court(foot_x, foot_y)

        pose_data = None
        if "pose" in p and p["pose"] is not None:
            pose_data = [
                {
                    "x": round(lm["x"], 1),
                    "y": round(lm["y"], 1),
                    "z": round(lm["z"], 4),
                    "vis": round(lm["visibility"], 2),
                }
                for lm in p["pose"]["landmarks"]
            ]

        player_states.append({
            "track_id": p.get("track_id", -1),
            "team": p.get("team", "unknown"),
            "bbox": list(bbox),
            "conf": round(p["conf"], 3),
            "foot_px": [round(foot_x, 1), round(foot_y, 1)],
            "court_pos": (
                [round(court_pos[0], 2), round(court_pos[1], 2)]
                if court_pos else None
            ),
            "on_court": on_court,
            "pose": pose_data,
        })

    ball_state = None
    if balls:
        b = balls[0]
        bx = (b["bbox"][0] + b["bbox"][2]) / 2.0
        by = (b["bbox"][1] + b["bbox"][3]) / 2.0
        court_ball = None
        if court_mapper and court_mapper.is_calibrated:
            court_ball = court_mapper.pixel_to_court(bx, by)
        ball_state = {
            "bbox": list(b["bbox"]),
            "conf": round(b["conf"], 3),
            "center_px": [round(bx, 1), round(by, 1)],
            "court_pos": (
                [round(court_ball[0], 2), round(court_ball[1], 2)]
                if court_ball else None
            ),
        }

    return {
        "frame_id": frame_id,
        "timestamp_s": round(timestamp_s, 4),
        "ball": ball_state,
        "players": player_states,
        "player_count": len(player_states),
        "on_court_count": sum(1 for p in player_states if p["on_court"]),
    }


class StateExporter:
    """Collects frame states and writes them to a JSON-lines file."""

    def __init__(self, output_path: Path):
        self._path = output_path
        self._fh = open(output_path, "w")
        self._count = 0

    def write(self, state: dict):
        self._fh.write(json.dumps(state, separators=(",", ":")) + "\n")
        self._count += 1

    def close(self):
        self._fh.close()
        print(f"  State exported: {self._count} frames → {self._path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
