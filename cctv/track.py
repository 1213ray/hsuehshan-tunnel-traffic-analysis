"""
雪隧 CCTV 即時車輛偵測、追蹤與分車道測速。

  YOLOv8 偵測車輛 → DeepSORT 追蹤 → 車輛底部中心點通過兩條測速線的時間差算出車速
  → 依車道分隔線判斷內／外側車道 → 統計最近幾分鐘各車道的平均車速，給出車道建議

用法（在 repo 根目錄）：
  python cctv/track.py --camera CCTV-N5-S-18.375-M --show
  python cctv/track.py --camera CCTV-N5-S-18.375-M --duration 600 --save-video runs/cctv/s18.mp4
  python cctv/track.py --camera CCTV-N5-S-18.375-M --source recording.mp4    # 離線影片
"""
import argparse
import csv
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from calibration import LaneAdvisor, SpeedMeter, load_camera

try:
    import truststore
    truststore.inject_into_ssl()    # 使用系統憑證庫：高公局 CCTV 網站的憑證鏈不在 requests 內建的 certifi 清單中
except ImportError:
    pass

VEHICLE_CLASSES = [2, 3, 5, 7]          # COCO：car, motorcycle, bus, truck
GREEN, YELLOW, RED, BLUE, WHITE = (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 128, 0), (255, 255, 255)


class MJPEGStream:
    """在背景執行緒讀取 MJPEG 串流，只保留最新一張畫面，處理較慢時自動跳過舊畫面。"""

    def __init__(self, url: str):
        self.url = url
        self.latest = None              # (frame, timestamp)
        self.lock = threading.Lock()
        self.stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stopped:
            try:
                with requests.get(self.url, stream=True, timeout=15) as r:
                    r.raise_for_status()
                    buf = b''
                    for chunk in r.iter_content(8192):
                        if self.stopped:
                            return
                        buf += chunk
                        start = buf.find(b'\xff\xd8')
                        end = buf.find(b'\xff\xd9', start + 2)
                        if start >= 0 and end >= 0:
                            jpg, buf = buf[start:end + 2], buf[end + 2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self.lock:
                                    self.latest = (frame, time.time())
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
                time.sleep(0.5)         # 伺服器約每 30 秒結束一次連線，立即重連
            except requests.RequestException as e:
                print(f'串流錯誤，5 秒後重新連線：{e}')
                time.sleep(5)

    def read(self):
        """等待下一張新畫面，回傳 (frame, 時間戳記)。"""
        while not self.stopped:
            with self.lock:
                item, self.latest = self.latest, None
            if item is not None:
                return item
            time.sleep(0.005)
        return None

    def stop(self):
        self.stopped = True


class VideoFile:
    """離線影片：逐張讀取，時間戳記由影格編號與 FPS 換算。"""

    def __init__(self, path: str, fps: float = 0):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.index = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.index += 1
        return frame, self.index / self.fps

    def stop(self):
        self.cap.release()


FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_label(img, text, x, y, color, scale=0.55):
    """在 (x, y) 左下角畫出黑底文字。"""
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = max(0, int(x)), max(h + 6, int(y))
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, color, 1, cv2.LINE_AA)


def draw(frame, cam, tracks, trails, labels, advisor, now, scale):
    out = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    s = lambda v: int(round(v * scale))
    w = out.shape[1]

    # 測速線 A（紅）、B（藍）與車道分隔線（白）
    for y, color, name in ((cam.gate_a_y, RED, 'A'), (cam.gate_b_y, BLUE, 'B')):
        cv2.line(out, (0, s(y)), (w, s(y)), color, 1)
        put_label(out, name, w - 22, s(y) - 2, color)
    if cam.divider is not None:
        a, b, c = cam.divider
        pts = [(s(-(b * y + c) / a), s(y)) for y in (cam.y_min, min(cam.y_max, frame.shape[0] - 1))]
        cv2.line(out, pts[0], pts[1], WHITE, 1)

    # 車輛：綠框、黃色軌跡、量到的車速
    for tid, (l, t, r, b) in tracks.items():
        trail = list(trails[tid])
        for p, q in zip(trail, trail[1:]):
            cv2.line(out, (s(p[0]), s(p[1])), (s(q[0]), s(q[1])), YELLOW, 2)
        cv2.rectangle(out, (s(l), s(t)), (s(r), s(b)), GREEN, 2)
        if tid in labels:
            put_label(out, labels[tid], s(l), s(t) - 3, GREEN)

    # 左下角：各車道平均車速與建議
    lines = [f'{lane}: {v:.0f} km/h (n={n})' for lane, (v, n) in advisor.lane_speeds(now).items()]
    lines.append(f'advice: {advisor.advice(now)}')
    for i, text in enumerate(lines):
        put_label(out, text, 6, out.shape[0] - 8 - 22 * (len(lines) - 1 - i), WHITE)
    return out


def main():
    ap = argparse.ArgumentParser(description='雪隧 CCTV 車輛追蹤與分車道測速')
    ap.add_argument('--camera', required=True, help='鏡頭編號，例如 CCTV-N5-S-18.375-M（對應 cctv/gates/ 的檔名）')
    ap.add_argument('--source', help='影像來源（網址或影片檔），預設使用設定檔裡的串流網址')
    ap.add_argument('--distance', type=float, help='覆蓋設定檔的測速線距離（公尺），用於重新校正')
    ap.add_argument('--fps', type=float, default=0, help='影片檔的實際 FPS（檔案內記錄的 FPS 不正確時使用）')
    ap.add_argument('--model', default='yolov8s.pt', help='YOLOv8 權重（只有 CPU 時可改用較快的 yolov8n.pt）')
    ap.add_argument('--conf', type=float, default=0.3, help='偵測信心門檻')
    ap.add_argument('--imgsz', type=int, default=640, help='YOLO 推論解析度')
    ap.add_argument('--max-age', type=int, default=15, help='DeepSORT：追蹤目標消失幾張畫面後刪除')
    ap.add_argument('--window', type=float, default=300, help='車道建議統計的時間窗（秒）')
    ap.add_argument('--duration', type=float, default=0, help='執行秒數，0 表示直到中斷')
    ap.add_argument('--show', action='store_true', help='開啟視窗顯示畫面（按 q 結束）')
    ap.add_argument('--save-video', help='另存標註後的影片（.mp4）')
    ap.add_argument('--log', help='量測紀錄 CSV，預設 runs/cctv/<鏡頭>_<時間>.csv')
    args = ap.parse_args()

    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from ultralytics import YOLO

    cam = load_camera(args.camera)
    if args.distance:
        cam.distance_m = args.distance
    src = args.source or cam.source
    stream = MJPEGStream(src) if src.startswith('http') else VideoFile(src, args.fps)
    model = YOLO(args.model)
    # 隧道內燈光變化大、近處車輛每張畫面位移也大：放寬外觀（cosine）與位置（IoU）的比對門檻，減少中途換編號
    tracker = DeepSort(max_age=args.max_age, n_init=2, max_cosine_distance=0.4, max_iou_distance=0.9,
                       embedder='mobilenet', bgr=True, embedder_gpu=torch.cuda.is_available())
    meter, advisor = SpeedMeter(cam), LaneAdvisor(window_s=args.window)

    log_path = Path(args.log or f'runs/cctv/{cam.camera_id}_{datetime.now():%Y%m%d_%H%M%S}.csv')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(log_file)
    writer.writerow(['time', 'camera', 'track_id', 'vehicle', 'direction', 'lane', 'speed_kmh'])

    trails = defaultdict(lambda: deque(maxlen=40))
    labels, last_seen = {}, {}
    frame_index = 0
    video = None
    scale = 2.0
    started = time.time()
    last_report, frames_at_report = started, 0
    print(f'{cam.camera_id}：{src}')

    try:
        while True:
            item = stream.read()
            if item is None:
                break
            frame, t = item
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=VEHICLE_CLASSES, verbose=False)[0]
            detections = [([x1, y1, x2 - x1, y2 - y1], float(conf), result.names[int(cls)])
                          for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.tolist(),
                                                                  result.boxes.conf.tolist(),
                                                                  result.boxes.cls.tolist())]
            active = {}
            for track in tracker.update_tracks(detections, frame=frame):
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                tid = str(track.track_id)
                box = track.to_ltrb(orig=True)      # 這一張畫面 YOLO 的原始偵測框（卡爾曼預測框會偏大、落後）
                if box is None:
                    continue
                l, top, r, b = box
                cx, by = (l + r) / 2, b             # 底部中心點：車輪接觸路面的位置
                active[tid] = (l, top, r, b)
                trails[tid].append((cx, by))
                m = meter.update(tid, track.get_det_class() or 'vehicle', cx, by, t)
                if m:
                    advisor.add(m)
                    labels[tid] = f'{m.speed_kmh:.0f} km/h'
                    stamp = datetime.fromtimestamp(t).isoformat(timespec='seconds') if t > 1e9 else f'{t:.2f}'
                    writer.writerow([stamp, cam.camera_id, tid, m.vehicle, m.direction, m.lane or '', f'{m.speed_kmh:.1f}'])
                    log_file.flush()
                    print(f'  #{tid} {m.vehicle} {m.direction} {m.lane or "-"} {m.speed_kmh:.1f} km/h')
            # 太久沒出現的追蹤編號：清掉軌跡、標籤與測速狀態
            frame_index += 1
            for tid in active:
                last_seen[tid] = frame_index
            for tid in [k for k, v in last_seen.items() if frame_index - v > 2 * args.max_age]:
                last_seen.pop(tid)
                trails.pop(tid, None)
                labels.pop(tid, None)
            meter.forget(set(last_seen))

            if args.show or args.save_video:
                annotated = draw(frame, cam, active, trails, labels, advisor, t, scale)
                if args.save_video:
                    if video is None:
                        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                        h, w = annotated.shape[:2]
                        video = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
                    video.write(annotated)
                if args.show:
                    cv2.imshow(cam.camera_id, annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            now = time.time()
            if now - last_report >= 30:
                speeds = ', '.join(f'{lane} {v:.0f} km/h (n={n})' for lane, (v, n) in advisor.lane_speeds(t).items())
                fps = (frame_index - frames_at_report) / (now - last_report)
                print(f'[{datetime.now():%H:%M:%S}] {fps:.1f} fps | {speeds or "尚無量測"} → {advisor.advice(t)}')
                if fps < 10 and src.startswith('http'):
                    print('  處理速度低於 10 fps，越線取樣間隔過長的車輛會被略過；建議使用 GPU 或調低 --imgsz')
                last_report, frames_at_report = now, frame_index
            if args.duration and now - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        log_file.close()
        if video is not None:
            video.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f'量測紀錄：{log_path}')


if __name__ == '__main__':
    main()
