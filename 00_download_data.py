"""
下載高公局交通資料庫的 VD（車輛偵測器）歷史動態資料 VDLive。

來源：https://tisvcloud.freeway.gov.tw/history/motc20/VD/YYYYMMDD/VDLive_HHMM.xml.gz
（每 15 分鐘一檔，全國 VD 都在同一檔內）

預設只下載 01_parse_xml.py 會用到的部分（週六日、07:00–22:45），
2024/12/22 – 2025/12/21 約 6,700 個檔案、1 GB；加上 --all-days 則下載全部日子。

用法：
    python 00_download_data.py
    python 00_download_data.py --start 2025-01-04 --end 2025-01-05
"""
import argparse
import datetime as dt
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = 'https://tisvcloud.freeway.gov.tw/history/motc20/VD/{date}/VDLive_{hhmm}.xml.gz'
OUT_DIR = 'data/vd_xml'
HOURS = range(7, 23)   # 與 01_parse_xml.py 的 TARGET_HOURS = (7, 22) 對應


def build_jobs(start, end, all_days, all_hours):
    jobs = []
    day = start
    while day <= end:
        if all_days or day.weekday() in (5, 6):
            hours = range(24) if all_hours else HOURS
            for h in hours:
                for m in (0, 15, 30, 45):
                    date = day.strftime('%Y%m%d')
                    hhmm = f'{h:02d}{m:02d}'
                    url = BASE_URL.format(date=date, hhmm=hhmm)
                    path = os.path.join(OUT_DIR, date, f'VDLive_{hhmm}.xml.gz')
                    jobs.append((url, path))
        day += dt.timedelta(days=1)
    return jobs


def download(job, retries=3):
    url, path = job
    if os.path.exists(path):
        return 'skip'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.replace(path + '.part', path)
            return 'ok'
        except Exception:
            time.sleep(2 ** attempt)
    return 'fail'


def main():
    parser = argparse.ArgumentParser(description='下載 VDLive 歷史資料')
    parser.add_argument('--start', default='2024-12-22', help='起始日期 YYYY-MM-DD')
    parser.add_argument('--end', default='2025-12-21', help='結束日期 YYYY-MM-DD（含）')
    parser.add_argument('--all-days', action='store_true', help='連平日也下載')
    parser.add_argument('--all-hours', action='store_true', help='下載 24 小時（預設只下載 07–22 點）')
    parser.add_argument('--workers', type=int, default=8, help='同時下載數')
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    jobs = build_jobs(start, end, args.all_days, args.all_hours)
    print(f'共 {len(jobs)} 個檔案，下載到 {OUT_DIR}/ ...')

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, status in enumerate(executor.map(download, jobs), 1):
            counts[status] += 1
            if i % 500 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)}  新下載 {counts["ok"]}、已存在 {counts["skip"]}、失敗 {counts["fail"]}')

    print('完成。請接著執行 01_parse_xml.py')


if __name__ == '__main__':
    main()
