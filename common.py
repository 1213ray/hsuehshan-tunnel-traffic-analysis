"""
共用設定與函式：資料路徑、車道對應、資料讀取、模型訓練。
02_train_model.py 與 03_visualization.py 都從這裡取資料與模型，確保兩者結果一致。
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ================= 參數設定區 =================
DATA_CSV = 'data/tunnel_data_final.csv.gz'   # 01_parse_xml.py 的輸出
FIG_DIR = 'figures'                          # 圖表輸出資料夾

# VD 資料只給 LaneID（0、1），沒有標示哪一條是內側。
# LaneID 1 有約 76% 的時間比 LaneID 0 快、平均車速高約 4 km/h、佔有率較低，
# 符合內側（超車道）的特性，因此將 LaneID 1 視為內側、LaneID 0 視為外側。
INNER_LANE = '1'
OUTER_LANE = '0'

SPEED_MARGIN = 3      # 內側比外側快超過 3 km/h，才標記為「建議走內側」
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ============================================

FEATURES = ['Occ_Inner', 'Occ_Outer']   # x1：內側佔有率、x2：外側佔有率 (%)


def load_data(path=DATA_CSV):
    """讀取 CSV，移除任一車道缺值的列，並換成內／外側欄位名稱與標籤 Target。"""
    df = pd.read_csv(path)
    raw_cols = [f'Speed_{OUTER_LANE}', f'Speed_{INNER_LANE}',
                f'Occupancy_{OUTER_LANE}', f'Occupancy_{INNER_LANE}']
    df[raw_cols] = df[raw_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=raw_cols).copy()

    df['Speed_Inner'] = df[f'Speed_{INNER_LANE}']
    df['Speed_Outer'] = df[f'Speed_{OUTER_LANE}']
    df['Occ_Inner'] = df[f'Occupancy_{INNER_LANE}']
    df['Occ_Outer'] = df[f'Occupancy_{OUTER_LANE}']
    df['Time'] = pd.to_datetime(df['Time'])

    # Target = 1：內側比外側快超過 SPEED_MARGIN（建議走內側）；0：建議走外側
    df['Target'] = (df['Speed_Inner'] > df['Speed_Outer'] + SPEED_MARGIN).astype(int)
    return df


def fit_model(df):
    """以 80/20 切分訓練 Logistic Regression，回傳模型與測試集。"""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df['Target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test
