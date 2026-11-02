"""
測速與車道判斷的單元測試：python -m pytest cctv/test_calibration.py
"""
import math

from calibration import Camera, LaneAdvisor, SpeedMeter, crossing_time, lane_name, load_camera


def make_camera(divider=None):
    # 與 CCTV-N5-S-18.375-M 相同的測速線：A 在 y=222、B 在 y=65，相距 50 公尺
    return Camera('test', '', gate_a_y=222, gate_b_y=65, distance_m=50.0, y_min=25, y_max=239, divider=divider)


def drive(meter, track_id, ys, fps=15.0, x=150.0, t0=0.0):
    """模擬一輛車的底部中心點依序經過 ys，回傳量測結果（沒有量到則為 None）。"""
    result = None
    for i, y in enumerate(ys):
        m = meter.update(track_id, 'car', x, y, t0 + i / fps)
        result = result or m
    return result


def test_crossing_time_interpolates():
    assert crossing_time(230, 1.0, 210, 2.0, 222) == 1.4
    assert crossing_time(230, 1.0, 225, 2.0, 222) is None      # 還沒越線
    assert crossing_time(222, 1.0, 222, 2.0, 222) is None      # 沒有移動


def test_speed_of_vehicle_moving_away():
    # 157 像素在 2 秒內走完（50 公尺 / 2 秒 = 90 km/h），畫面每 1/15 秒一張
    ys = [235 - (235 - 50) * i / 35 for i in range(36)]        # y 從 235 均速降到 50
    seconds_between_gates = (222 - 65) / (235 - 50) * 35 / 15
    m = drive(SpeedMeter(make_camera()), '1', ys)
    assert m.direction == 'away'
    assert math.isclose(m.speed_kmh, 50 / seconds_between_gates * 3.6, rel_tol=1e-6)


def test_vehicle_moving_toward_camera():
    m = drive(SpeedMeter(make_camera()), '1', [40 + 6 * i for i in range(34)])
    assert m.direction == 'toward' and m.speed_kmh > 0


def test_each_vehicle_is_measured_once():
    meter = SpeedMeter(make_camera())
    first = drive(meter, '1', [235 - 6 * i for i in range(32)])
    again = drive(meter, '1', [235 - 6 * i for i in range(32)], t0=10)
    assert first is not None and again is None


def test_out_of_band_points_do_not_count():
    meter = SpeedMeter(make_camera())
    # 從有效範圍外（y=10）直接跳到 y=230，不能算成越過 B 線
    assert drive(meter, '1', [10, 230, 228]) is None


def test_lane_names():
    assert lane_name('left', 'away') == 'inner'
    assert lane_name('right', 'away') == 'outer'
    assert lane_name('left', 'toward') == 'outer'
    assert lane_name(None, 'away') is None


def test_real_lane_divider():
    cam = load_camera('CCTV-N5-S-18.375-M')
    a, b, c = cam.divider
    x_on_line = -(b * 200 + c) / a
    assert cam.side(x_on_line - 20, 200) == 'left'
    assert cam.side(x_on_line + 20, 200) == 'right'


def test_both_config_schemas_load():
    north = load_camera('CCTV-N5-N-21.014-M')      # 使用 roi
    south = load_camera('CCTV-N5-S-18.375-M')      # 使用 band
    assert north.y_min == 52 and north.divider is None
    assert south.y_min == 25 and south.divider is not None


def test_lane_advice():
    advisor = LaneAdvisor(window_s=300, margin=3)
    meter = SpeedMeter(make_camera(divider=(1.0, 0.0, -150.0)))     # 分隔線 x = 150
    for i in range(5):
        m = drive(meter, f'in{i}', [235 - 8 * i2 for i2 in range(25)], x=100, t0=i)    # 內側（左）較快
        advisor.add(m)
        m = drive(meter, f'out{i}', [235 - 5 * i2 for i2 in range(40)], x=200, t0=i)   # 外側（右）較慢
        advisor.add(m)
    speeds = advisor.lane_speeds(now=10)
    assert speeds['inner'][0] > speeds['outer'][0]
    assert advisor.advice(now=10) == 'take the INNER lane'


def test_long_sampling_gap_is_rejected():
    meter = SpeedMeter(make_camera(), max_gap=0.35)
    # 每 0.5 秒才取樣一次：內插不可靠，不應產生量測
    assert drive(meter, '1', [235 - 20 * i for i in range(10)], fps=2) is None


def test_gap_away_from_lines_is_tolerated():
    meter = SpeedMeter(make_camera(), max_gap=0.35)
    # 在兩條線之間漏偵測 0.6 秒（y=150 → 120），越線前後的取樣都很密，仍應量得到
    ys = [235 - 6 * i for i in range(15)] + [120 - 6 * i for i in range(15)]
    ts = [i / 15 for i in range(15)] + [14 / 15 + 0.6 + i / 15 for i in range(15)]
    result = None
    for y, t in zip(ys, ts):
        result = result or meter.update('1', 'car', 150, y, t)
    assert result is not None
