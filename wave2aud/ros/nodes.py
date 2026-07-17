"""ROS2 nodes: treat sonification as a robot perception stream.

Topics / message contract (kept portable with std_msgs so no custom .msg build
is required):

* Input  ``~/wave``  : ``std_msgs/Float32MultiArray`` -- raw sensor samples.
  For complex data (radar IQ / heterodyned ultrasound) interleave
  ``[re0, im0, re1, im1, ...]`` and set parameter ``is_complex=true``.
* Input  ``~/wave_meta`` : ``std_msgs/String`` (JSON) -- optional per-message
  side channels (range_m, azimuth_deg, event_times, temperature_k, ...).
* Output ``~/audio`` : ``std_msgs/Float32MultiArray`` -- stereo interleaved
  ``[L0, R0, L1, R1, ...]`` cross-faded audio blocks.
* Output ``~/features`` : ``std_msgs/String`` (JSON) -- the extracted
  perceptual features, so other nodes can react (e.g. steer toward a bright,
  fast-onset gamma source).

Parameters: ``wave_type``, ``sample_rate``, ``carrier``, ``is_complex``,
``audio_fs``.

Run (on a machine with ROS2)::

    ros2 run wave2aud sonifier --ros-args -p wave_type:=radar -p sample_rate:=8000.0
    # or from python:
    python -m wave2aud.ros.nodes sonifier
"""
from __future__ import annotations

import json

import numpy as np

from ..realtime import RealtimeSonifier
from ..waves import WaveSample


class WaveBridge:
    """ROS-independent core: bytes/arrays in -> audio + features out.

    Backed by the stateful :class:`~wave2aud.realtime.RealtimeSonifier`, so
    successive blocks join seamlessly. Unit-testable without ROS2; the nodes
    below are thin wrappers over this.
    """

    def __init__(self, wave_type: str, sample_rate: float, carrier=None,
                 is_complex: bool = False, audio_fs: float = 44100.0):
        self.wave_type = wave_type
        self.sample_rate = float(sample_rate)
        self.carrier = carrier
        self.is_complex = bool(is_complex)
        self.engine = RealtimeSonifier(fs=audio_fs)

    def _to_sample(self, flat, meta) -> WaveSample:
        arr = np.asarray(flat, dtype=float)
        if self.is_complex:
            arr = arr[::2] + 1j * arr[1::2]
        return WaveSample(arr, self.sample_rate, self.wave_type, self.carrier, meta=meta or {})

    def process(self, flat, meta=None):
        sample = self._to_sample(flat, meta)
        audio = self.engine.process(sample)                 # [n, 2], seamless
        interleaved = audio.reshape(-1).astype(np.float32)  # [L0,R0,...]
        feats = self.engine.last_features
        fdict = {
            "loudness": feats.loudness, "brightness": feats.brightness,
            "centroid_hz": feats.centroid_hz, "onset_rate": feats.onset_rate,
            "tempo_bpm": feats.tempo_bpm, "flatness": feats.flatness,
        }
        return interleaved, fdict


# ---------------------------------------------------------------------------
# ROS2 node factories (only usable when rclpy is present)
# ---------------------------------------------------------------------------
def make_sonifier_node():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    class SonifierNode(Node):
        def __init__(self):
            super().__init__("wave2aud_sonifier")
            self.declare_parameter("wave_type", "radar")
            self.declare_parameter("sample_rate", 8000.0)
            self.declare_parameter("carrier", 0.0)
            self.declare_parameter("is_complex", False)
            self.declare_parameter("audio_fs", 44100.0)
            gp = self.get_parameter
            carrier = gp("carrier").value or None
            self.bridge = WaveBridge(
                gp("wave_type").value, gp("sample_rate").value, carrier,
                gp("is_complex").value, gp("audio_fs").value,
            )
            self._meta = {}
            self.audio_pub = self.create_publisher(Float32MultiArray, "~/audio", 10)
            self.feat_pub = self.create_publisher(String, "~/features", 10)
            self.create_subscription(Float32MultiArray, "~/wave", self._on_wave, 10)
            self.create_subscription(String, "~/wave_meta", self._on_meta, 10)
            self.get_logger().info(f"wave2aud sonifier up for {gp('wave_type').value}")

        def _on_meta(self, msg):
            try:
                self._meta = json.loads(msg.data)
            except Exception:
                self._meta = {}

        def _on_wave(self, msg):
            interleaved, fdict = self.bridge.process(msg.data, self._meta)
            am = Float32MultiArray(); am.data = interleaved.tolist()
            self.audio_pub.publish(am)
            sm = String(); sm.data = json.dumps(fdict)
            self.feat_pub.publish(sm)

    return SonifierNode()


def make_wave_publisher_node(wave_type: str = "radar", period: float = 1.5):
    """A demo publisher that streams simulated waves onto ``~/wave``."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    from ..sources import SimulatedSource

    class WavePublisher(Node):
        def __init__(self):
            super().__init__("wave2aud_wave_publisher")
            self.src = SimulatedSource(wave_type)
            self.wave_pub = self.create_publisher(Float32MultiArray, "~/wave", 10)
            self.meta_pub = self.create_publisher(String, "~/wave_meta", 10)
            self.create_timer(period, self._tick)

        def _tick(self):
            s = self.src.read()
            arr = s.data
            if np.iscomplexobj(arr):
                flat = np.empty(arr.size * 2, dtype=np.float32)
                flat[::2] = np.real(arr); flat[1::2] = np.imag(arr)
            else:
                flat = np.real(arr).astype(np.float32)
            m = Float32MultiArray(); m.data = flat.tolist()
            self.wave_pub.publish(m)
            meta = {k: v for k, v in s.meta.items() if np.isscalar(v)}
            sm = String(); sm.data = json.dumps(meta)
            self.meta_pub.publish(sm)

    return WavePublisher()


def main(argv=None):  # pragma: no cover - requires ROS2
    import sys
    import rclpy

    which = (argv or sys.argv[1:] or ["sonifier"])[0]
    rclpy.init()
    node = make_wave_publisher_node() if which == "publisher" else make_sonifier_node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
