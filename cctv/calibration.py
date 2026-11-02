"""
鏡頭標定設定的讀取，以及測速、車道判斷的幾何計算。

每支鏡頭有兩個設定檔（畫面 352×240 像素）：
  gates/<鏡頭>.json   gateA_y、gateB_y：兩條水平測速線的 y 座標（A 在畫面下方、B 在上方）
                      distance_m：兩條線在路面上的實際距離（公尺）
                      band（部分檔案叫 roi）：偵測結果有效的 y 範圍
  lanes/<鏡頭>.json   line_abc：車道分隔線 A·x + B·y + C = 0（只有南向 6 支）

車速 = distance_m ÷ 車輛通過兩條測速線的時間差。
車道以車輛底部中心點在分隔線左側或右側判斷；台灣靠右行駛，車輛遠離鏡頭時，畫面左側是內側車道。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

CCTV_DIR = Path(__file__).resolve().parent


@dataclass
class Camera:
    camera_id: str
    source: str                 # 影像串流網址
    gate_a_y: float
    gate_b_y: float
    distance_m: float
    y_min: float                # 有效偵測範圍
    y_max: float
    divider: Optional[Tuple[float, float, float]] = None   # 車道分隔線 (A, B, C)

    def in_band(self, y: float) -> bool:
        return self.y_min <= y <= self.y_max

    def side(self, x: float, y: float) -> Optional[str]:
        """點在車道分隔線的畫面左側或右側；沒有車道設定時回傳 None。"""
        if self.divider is None:
            return None
        a, b, c = self.divider
        divider_x = -(b * y + c) / a        # 分隔線在這個高度的 x 座標
        return 'left' if x < divider_x else 'right'


def load_camera(camera_id: str, cctv_dir: Path = CCTV_DIR) -> Camera:
    """讀取 gates/ 與 lanes/ 中該鏡頭的設定。"""
    gate = json.loads((cctv_dir / 'gates' / f'{camera_id}.json').read_text(encoding='utf-8'))
    band = gate.get('band') or gate.get('roi') or {'y_min': 0, 'y_max': float('inf')}

    divider = None
    lane_file = cctv_dir / 'lanes' / f'{camera_id}.json'
    if lane_file.exists():
        abc = json.loads(lane_file.read_text(encoding='utf-8'))['line_abc']
        divider = (abc['A'], abc['B'], abc['C'])

    return Camera(camera_id, gate.get('source_hint', ''), gate['gateA_y'], gate['gateB_y'],
                  gate['distance_m'], band['y_min'], band['y_max'], divider)


def crossing_time(y0: float, t0: float, y1: float, t1: float, line_y: float) -> Optional[float]:
    """從 (y0, t0) 移動到 (y1, t1) 時若越過 line_y，以線性內插回傳越線時間，否則回傳 None。"""
    if y0 == y1 or (y0 - line_y) * (y1 - line_y) > 0:
        return None
    return t0 + (line_y - y0) / (y1 - y0) * (t1 - t0)


def lane_name(side: Optional[str], direction: str) -> Optional[str]:
    """畫面左右 → 內／外側車道。遠離鏡頭（away）時畫面左側是內側，朝向鏡頭時相反。"""
    if side is None:
        return None
    return 'inner' if (side == 'left') == (direction == 'away') else 'outer'


@dataclass
class Measurement:
    time: float
    track_id: str
    vehicle: str
    direction: str              # away：遠離鏡頭（先過 A 線）；toward：朝向鏡頭
    lane: Optional[str]         # inner / outer；沒有車道設定時為 None
    speed_kmh: float


@dataclass
class _TrackState:
    last: Optional[Tuple[float, float, float]] = None     # 上一次的 (x, y, t)
    time_a: Optional[float] = None
    time_b: Optional[float] = None
    side: Optional[str] = None                            # 通過 A 線（畫面下方、車輛較大）時的左右
    done: bool = False


class SpeedMeter:
    """
    依各追蹤編號的底部中心點軌跡，量測通過兩條測速線的車速。每輛車只量一次。

    越線時間以前後兩次取樣線性內插。畫面的 y 座標與路面距離不是線性關係（近大遠小），
    兩次取樣相隔太久時內插誤差很大，因此超過 max_gap 秒就放棄這次量測。
    """

    MIN_KMH, MAX_KMH = 20.0, 160.0      # 超出範圍視為追蹤錯誤

    def __init__(self, camera: Camera, max_gap: float = 0.35):
        self.camera = camera
        self.max_gap = max_gap
        self.tracks: Dict[str, _TrackState] = {}

    def update(self, track_id: str, vehicle: str, x: float, y: float, t: float) -> Optional[Measurement]:
        cam = self.camera
        state = self.tracks.setdefault(track_id, _TrackState())
        if not cam.in_band(y):
            state.last = None           # 離開有效範圍就重新開始，避免跨區跳動被當成越線
            return None
        if state.done:
            return None

        if state.last is not None:
            x0, y0, t0 = state.last
            long_gap = t - t0 > self.max_gap
            tc_a = crossing_time(y0, t0, y, t, cam.gate_a_y) if state.time_a is None else None
            tc_b = crossing_time(y0, t0, y, t, cam.gate_b_y) if state.time_b is None else None
            if long_gap and (tc_a is not None or tc_b is not None):
                state.done = True       # 越線發生在一段很長的漏偵測之間，越線時間不可靠
                return None
            if tc_a is not None:
                state.time_a = tc_a
                xa = x0 + (x - x0) * (cam.gate_a_y - y0) / (y - y0)
                state.side = cam.side(xa, cam.gate_a_y)
            if tc_b is not None:
                state.time_b = tc_b
        state.last = (x, y, t)

        if state.time_a is None or state.time_b is None:
            return None
        state.done = True
        dt = abs(state.time_b - state.time_a)
        if dt <= 0:
            return None
        speed = cam.distance_m / dt * 3.6
        if not self.MIN_KMH <= speed <= self.MAX_KMH:
            return None
        direction = 'away' if state.time_a < state.time_b else 'toward'
        return Measurement(t, track_id, vehicle, direction, lane_name(state.side, direction), speed)

    def forget(self, active_ids):
        """移除已消失的追蹤，避免記憶體持續增加。"""
        for tid in list(self.tracks):
            if tid not in active_ids:
                del self.tracks[tid]


@dataclass
class LaneAdvisor:
    """統計最近 window_s 秒內各車道的平均車速，快超過 margin km/h 的車道即為建議車道。"""
    window_s: float = 300.0
    margin: float = 3.0                 # 與 VD 模型的標籤定義相同
    records: list = field(default_factory=list)

    def add(self, m: Measurement):
        if m.lane is not None:
            self.records.append(m)

    def lane_speeds(self, now: float) -> Dict[str, Tuple[float, int]]:
        self.records = [m for m in self.records if now - m.time <= self.window_s]
        out = {}
        for lane in ('inner', 'outer'):
            speeds = [m.speed_kmh for m in self.records if m.lane == lane]
            if speeds:
                out[lane] = (sum(speeds) / len(speeds), len(speeds))
        return out

    def advice(self, now: float) -> str:
        stats = self.lane_speeds(now)
        if len(stats) < 2:
            return 'not enough data'
        inner, outer = stats['inner'][0], stats['outer'][0]
        if inner > outer + self.margin:
            return 'take the INNER lane'
        if outer > inner + self.margin:
            return 'take the OUTER lane'
        return 'either lane'
