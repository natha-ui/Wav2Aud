"""ROS2 integration for wave2aud.

Import is safe even without ROS2 installed: the node classes are only defined
when ``rclpy`` is importable. Use :func:`available` to check.
"""
from __future__ import annotations


def available() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except Exception:
        return False


from .nodes import make_sonifier_node, make_wave_publisher_node, WaveBridge  # noqa: E402

__all__ = ["available", "make_sonifier_node", "make_wave_publisher_node", "WaveBridge"]
