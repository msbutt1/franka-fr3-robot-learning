"""Pure geometry helpers for FR3 grid and basket exclusion."""

from __future__ import annotations

import numpy as np


BASKET_CORNER_LABELS = [
    "basket_front_left",
    "basket_front_right",
    "basket_back_right",
    "basket_back_left",
]


def basket_polygon_from_points(points: dict) -> np.ndarray | None:
    if not all(label in points for label in BASKET_CORNER_LABELS):
        return None
    return np.array([points[label][:2] for label in BASKET_CORNER_LABELS], dtype=float)


def _point_in_polygon_xy(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    x, y = point_xy
    inside = False
    n = len(polygon_xy)
    for i in range(n):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def _point_segment_distance_xy(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    ab = b_xy - a_xy
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return float(np.linalg.norm(point_xy - a_xy))
    t = float(np.clip(np.dot(point_xy - a_xy, ab) / denom, 0.0, 1.0))
    closest = a_xy + t * ab
    return float(np.linalg.norm(point_xy - closest))


def inside_basket_exclusion(
    xyz: np.ndarray,
    pad_center: np.ndarray,
    basket_margin: float,
    basket_w: float,
    basket_h: float,
    basket_polygon_xy: np.ndarray | None,
) -> bool:
    point_xy = np.array(xyz[:2], dtype=float)
    if basket_polygon_xy is not None:
        if _point_in_polygon_xy(point_xy, basket_polygon_xy):
            return True
        distances = [
            _point_segment_distance_xy(point_xy, basket_polygon_xy[i], basket_polygon_xy[(i + 1) % len(basket_polygon_xy)])
            for i in range(len(basket_polygon_xy))
        ]
        return min(distances) < basket_margin

    half_w = basket_w / 2 + basket_margin
    half_h = basket_h / 2 + basket_margin
    return abs(xyz[0] - pad_center[0]) < half_w and abs(xyz[1] - pad_center[1]) < half_h
