"""
解析 VDLive XML，篩出雪山隧道內的車道資料，輸出 data/tunnel_data_final.csv.gz。

篩選條件：週六日、07:00–22:59、國道 5 號里程 15.2k–28.4k（雪隧範圍，南北向皆含）。
輸出欄位：Time、VDID、Occupancy_0/1、Speed_0/1（LaneID 0、1 兩條車道）。
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import time
import gzip

# ================= 參數設定區 =================
XML_ROOT_FOLDER = 'data/vd_xml'                  # 00_download_data.py 的下載位置
OUTPUT_CSV = 'data/tunnel_data_final.csv.gz'     # 輸出的檔案名稱（.gz 會自動壓縮）
TARGET_HOURS = (7, 22)                # 鎖定時段：早上7點到晚上10點
MIN_MILEAGE = 15.2                    # 雪隧起點里程 (k)
MAX_MILEAGE = 28.4                    # 雪隧終點里程 (k)
# ============================================

ns = {'ns': 'http://traffic.transportdata.tw/standard/traffic/schema/'}

def process_single_file(file_path):
    """
    單一檔案處理函數 (由 CPU 核心平行呼叫)
    包含：日期過濾、時段過濾、里程過濾
    """
    local_data = []
    
    try:
        # --- 1. 日期過濾 (從資料夾名稱判斷，不解壓直接跳過平日) ---
        # 假設路徑結構為 data_xml/20241222/xxx.xml.gz
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        try:
            folder_date = datetime.strptime(parent_dir_name, "%Y%m%d")
            # 0=週一 ... 5=週六, 6=週日。非六日則回傳空列表
            if folder_date.weekday() not in [5, 6]:
                return []
        except ValueError:
            pass # 如果資料夾名稱不是日期，就忽略這個檢查，繼續往下讀

        # --- 2. 解壓縮並解析 XML ---
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            
            # --- 3. 時段過濾 ---
            update_time_str = root.find('ns:UpdateTime', ns).text
            # 格式: 2024-12-22T14:15:00+08:00
            time_part = update_time_str.split('+')[0]
            dt_obj = datetime.strptime(time_part, "%Y-%m-%dT%H:%M:%S")
            
            if not (TARGET_HOURS[0] <= dt_obj.hour <= TARGET_HOURS[1]):
                return []

            # --- 4. 空間過濾 (雪隧里程) ---
            for vd in root.findall('.//ns:VDLive', ns):
                vdid = vd.find('ns:VDID', ns).text
                
                # 必須是國道五號
                if 'VD-N5' not in vdid:
                    continue
                
                # 解析里程數 (例如 VD-N5-S-015.700-M-LOOP)
                try:
                    parts = vdid.split('-')
                    # 嘗試找出數字部分的里程 (通常在 index 3)
                    mileage = float(parts[3])
                    
                    # 如果不在雪隧範圍內，跳過此 VD
                    if not (MIN_MILEAGE <= mileage <= MAX_MILEAGE):
                        continue
                except (IndexError, ValueError):
                    continue # 無法解析里程則跳過

                # 提取車道數據
                for lane in vd.findall('.//ns:Lane', ns):
                    lane_id = lane.find('ns:LaneID', ns).text
                    speed_text = lane.find('ns:Speed', ns).text
                    occ_text = lane.find('ns:Occupancy', ns).text
                    
                    speed = float(speed_text) if speed_text else 0.0
                    occupancy = float(occ_text) if occ_text else 0.0
                    
                    # 排除靜止或無效數據
                    if speed > 0:
                        local_data.append({
                            'Time': update_time_str,
                            'VDID': vdid,
                            'LaneID': lane_id,
                            'Speed': speed,
                            'Occupancy': occupancy
                        })
                        
    except Exception:
        pass # 檔案損壞直接跳過
        
    return local_data

def main():
    print(f"【資料前處理】開始搜尋 {XML_ROOT_FOLDER} 下的所有 .xml.gz 檔案...")
    # 遞迴搜尋所有子資料夾
    search_pattern = os.path.join(XML_ROOT_FOLDER, '**', '*.xml.gz')
    xml_files = glob.glob(search_pattern, recursive=True)
    
    total_files = len(xml_files)
    if total_files == 0:
        print("錯誤：找不到任何 .xml.gz 檔案！請先執行 00_download_data.py 下載資料。")
        return

    print(f"共找到 {total_files} 個檔案。")
    print(f"啟動 CPU 多核心運算...")
    print(f"過濾條件：[週六日] + [{TARGET_HOURS[0]}:00-{TARGET_HOURS[1]}:00] + [雪隧里程 {MIN_MILEAGE}-{MAX_MILEAGE}k]")
    
    start_time = time.time()
    all_results = []

    # 使用所有 CPU 核心平行處理
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_file, xml_files))
        for res in results:
            if res:
                all_results.extend(res)

    end_time = time.time()
    print(f"處理完成！耗時 {end_time - start_time:.2f} 秒。")
    print(f"篩選後剩餘 {len(all_results)} 筆有效數據。")

    if len(all_results) == 0:
        print("警告：沒有抓到資料，請檢查 XML 內容或過濾條件。")
        return

    # 轉檔 (Pivot Table)
    print("正在轉置資料表格 (將 LaneID 轉為欄位)...")
    raw_df = pd.DataFrame(all_results)
    
    # 將長表格轉為寬表格：每個時間點+VDID 為一列，內外車道並排
    df_pivot = raw_df.pivot_table(
        index=['Time', 'VDID'], 
        columns='LaneID', 
        values=['Speed', 'Occupancy']
    ).reset_index()
    
    # 欄位扁平化與重新命名 (例如 Speed_1, Occupancy_1)
    df_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_pivot.columns.values]
    
    df_pivot.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"成功產出 CSV 檔案：{OUTPUT_CSV}")
    print(f"欄位範例：{df_pivot.columns.tolist()}")
    print("-" * 50)
    print("請接著執行 02_train_model.py")

if __name__ == '__main__':
    main()