# Hsuehshan Tunnel Lane Advisor

**English** | [繁體中文](README.zh-TW.md)

Which lane should you take into the Hsuehshan Tunnel? Lane changes are prohibited inside the 12.9 km tunnel on National Freeway 5 in Taiwan, so a driver picks a lane once, at the entrance. This project looks at the tunnel's traffic from two sources:

1. **Live CCTV.** Vehicles in the tunnel's public camera feeds are detected with YOLOv8 and tracked with DeepSORT. Virtual gates measure each vehicle's speed, and a divider line assigns it to a lane.
2. **One year of vehicle detector (VD) data**, 258,399 records. These train a logistic regression that predicts, from the current occupancy of the two lanes, whether the inner lane will be faster.

December 2025. The VD analysis started as a *Computational Mathematics* course assignment.

<p align="center">
  <img src="docs/cctv_tracking.png" width="900" alt="Vehicle detection and tracking on six tunnel cameras">
</p>

## Part 1 · CCTV vehicle tracking

[`cctv/track.py`](cctv/track.py) reads a tunnel camera's live MJPEG stream (or a recorded video), detects vehicles with YOLOv8, tracks them with DeepSORT, measures each vehicle's speed between two virtual lines and assigns it to the inner or outer lane. It keeps the average speed of each lane over the last five minutes and recommends the lane that is faster by more than 3 km/h, the same margin the VD model uses.

<p align="center">
  <img src="docs/cctv_speed_demo.jpg" width="900" alt="Tracked vehicles with measured speeds and lane advice">
</p>

<sub>Red line A and blue line B are the speed gates, the thin white line is the lane divider, and yellow lines are tracks. The bottom-left panel shows per-lane averages and the current advice.</sub>

**How a speed is measured** ([`cctv/calibration.py`](cctv/calibration.py))

1. The reference point of a vehicle is the bottom center of its YOLO box, where the wheels meet the road.
2. The time the point crosses each gate is interpolated linearly between the two frames around the crossing. Image height is not proportional to road distance (far away, a metre spans fewer pixels), so a crossing is only used when those two frames are at most 0.35 s apart.
3. Speed = `distance_m` ÷ (time at gate B − time at gate A).
4. The lane is the side of the divider the vehicle is on when it crosses gate A, the gate nearer the camera. Taiwan drives on the right, so for vehicles moving away from the camera the left side is the inner lane.

**Calibration.** Twelve cameras inside the tunnel (352×240 pixels, 6 northbound and 6 southbound, km 16.9 – 26.4) were calibrated by hand. Each camera has a JSON file in [`cctv/`](cctv/):

| File | Fields | Use |
|---|---|---|
| `gates/<camera>.json` (12 cameras) | `gateA_y`, `gateB_y`: y-coordinates of two horizontal virtual lines; `distance_m`: real distance between them (15–50 m); `band` (named `roi` in five northbound files): the y-range where detections count | Speed measurement |
| `lanes/<camera>.json` (6 southbound cameras) | `divider.p1`, `divider.p2`: two points on the lane divider; `line_abc`: the line A·x + B·y + C = 0 through them, with (A, B) a unit normal | Lane assignment |

The camera ID encodes direction and position, for example `CCTV-N5-S-18.375-M` = National Freeway 5, southbound, km 18.375. Each file also keeps the stream URL (`source_hint`) and the calibration time.

**Test run.** CCTV-N5-S-18.375-M was recorded for 148 seconds on 2 November 2026 from 22:15, then processed frame by frame. That gave 18 measured vehicles (16 cars, 2 buses), which can be compared with the nearest VD detector, 63 m away, for the same 5-minute interval:

| | Inner lane | Outer lane |
|---|---|---|
| Vehicles measured by CCTV | 11 | 7 |
| CCTV mean speed | 116.5 km/h | 114.0 km/h |
| VD-N5-S-18.312 (22:15) | 95 km/h | 84 km/h |

The CCTV speeds run about 20–35% higher than the detector's. The hand-measured gate distance for this camera (50 m) is therefore probably too long; `--distance` overrides it for recalibration.

### Notes

- The original tracking script was lost. `cctv/track.py` is a November 2026 rewrite built on the original calibration files.
- As of 2 November 2026, 7 of the 12 stream URLs work. N-25.335, S-16.874, S-21.035, S-23.896 and S-25.298 return 404, and their camera IDs probably changed.
- Speeds are only reliable when frames are processed in real time (about 14 fps). A GPU handles that easily. On a CPU, use `--model yolov8n.pt`, or record first and process the video with `--source`.
- The CCTV site's certificate chain is not in the certificate list that `requests` ships with, so the script uses the operating system's trust store through `truststore`.

## Part 2 · Lane choice model from VD data

<p align="center">
  <img src="figures/decision_boundary.png" width="560" alt="Decision boundary of the lane choice model">
</p>

### Data

| | |
|---|---|
| Source | Freeway Bureau traffic database, VD live data (`VDLive`), [tisvcloud.freeway.gov.tw](https://tisvcloud.freeway.gov.tw/history/motc20/VD/) |
| Period | 2024-12-22 – 2025-12-21, weekends only (105 days), 07:00–22:59, one record every 15 minutes |
| Location | National Freeway 5, km 15.2 – 28.4 (the tunnel), both directions, 39 detectors |
| Size | 258,399 rows after parsing; 256,801 rows with both lanes present |
| Fields | `Time`, `VDID`, `Occupancy_0/1` (%), `Speed_0/1` (km/h) for LaneID 0 and 1 |

The parsed dataset is included as [`data/tunnel_data_final.csv.gz`](data/tunnel_data_final.csv.gz) (2 MB). The raw XML (about 1 GB for the weekend hours used) can be downloaded again with `00_download_data.py`; re-parsing it reproduces the CSV exactly.

**Which lane is the inner one?** The VD feed numbers lanes 0 and 1 but does not say which is the inner lane. LaneID 1 is faster 76% of the time (mean 74.6 vs 70.2 km/h) and has lower occupancy (16.3% vs 18.1%), as expected of the inner (passing) lane, so LaneID 1 is treated as the inner lane. This mapping is set in one place, `common.py`.

<p align="center">
  <img src="figures/lane_speed_by_hour.png" width="680" alt="Mean speed by hour for each lane">
</p>

### Method

- **Label.** `Target = 1` (take the inner lane) when the inner lane is faster than the outer lane by more than 3 km/h; otherwise `Target = 0`.
- **Features.** Inner-lane occupancy *x*₁ and outer-lane occupancy *x*₂ (%), which a driver could see on a live traffic map.
- **Model.** Logistic regression, 80/20 random train/test split (`random_state=42`):

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2, \qquad P(\text{inner lane}) = \frac{1}{1 + e^{-z}}$$

### Results

| Metric | Value |
|---|---|
| Test accuracy | **73.3%** |
| Majority-class baseline | 56.4% |
| Recall, inner / outer | 0.838 / 0.597 |
| F1, inner / outer | 0.780 / 0.661 |

| Coefficient | Value | Meaning |
|---|---|---|
| β₀ (intercept) | 1.1096 | With both lanes empty, P(inner) = 75.2% |
| β₁ (inner occupancy) | −0.2155 | Each +1% inner occupancy multiplies the odds of "inner" by 0.806 |
| β₂ (outer occupancy) | +0.1464 | Each +1% outer occupancy multiplies the odds of "inner" by 1.158 |

The decision boundary (P = 0.5) is **x₂ = −7.58 + 1.47·x₁**. Take the inner lane unless its occupancy exceeds roughly (x₂ + 7.6) / 1.47. For example, with the outer lane at 20%, the inner lane stays the better choice up to about 19% occupancy.

### Limitations

- The inner/outer mapping is inferred from the data, not from documentation.
- The random split mixes neighbouring 15-minute records and detectors between train and test, so the accuracy is probably optimistic. A split by date would be a stricter test.
- Two occupancy values explain only part of the speed difference. Hour of day, direction and detector position are obvious next features, and so are per-lane speeds measured from the CCTV feeds.

## Run it

**CCTV tracking** (needs `ultralytics`, `deep-sort-realtime`, `opencv-python`; a GPU is recommended)

```bash
pip install -r cctv/requirements.txt

python cctv/track.py --camera CCTV-N5-S-18.375-M --show                    # live, with a video window (q to quit)
python cctv/track.py --camera CCTV-N5-S-18.375-M --duration 600 --save-video runs/cctv/s18.mp4   # 10 minutes, save annotated video
python cctv/track.py --camera CCTV-N5-S-18.375-M --source recording.mp4 --fps 14.1   # recorded video
python -m pytest cctv/test_calibration.py                                   # unit tests for gates and lanes
```

Every measurement is appended to `runs/cctv/<camera>_<time>.csv` (time, track ID, vehicle type, direction, lane, speed).

**VD model**

```bash
pip install -r requirements.txt

python 02_train_model.py      # train and print metrics (uses the included CSV)
python 03_visualization.py    # write figures/*.png

# Optional: rebuild the dataset from raw XML
python 00_download_data.py    # weekends 07:00–22:45, ~6,700 files → data/vd_xml/
python 01_parse_xml.py        # → data/tunnel_data_final.csv.gz
```

## Project structure

```
├── cctv/
│   ├── track.py            live detection, tracking, speed measurement and lane advice
│   ├── calibration.py      camera configs, gate crossing, lane side, per-lane statistics
│   ├── test_calibration.py unit tests
│   ├── gates/              speed-measurement gates for 12 tunnel cameras
│   └── lanes/              lane dividers for 6 southbound cameras
├── 00_download_data.py     download VDLive XML from the Freeway Bureau archive
├── 01_parse_xml.py         filter tunnel detectors and lanes, write the CSV (multi-process)
├── 02_train_model.py       logistic regression, metrics and interpretation
├── 03_visualization.py     speed-by-hour and decision-boundary figures
├── common.py               paths, lane mapping, data loading, model fitting
├── data/
│   └── tunnel_data_final.csv.gz
├── figures/
└── docs/
```
