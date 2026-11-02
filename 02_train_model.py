"""
以內、外側車道佔有率預測「該走哪一條車道」的 Logistic Regression 模型。

    z = β0 + β1·x1 + β2·x2        x1：內側佔有率、x2：外側佔有率 (%)
    P(建議走內側) = 1 / (1 + e^(−z))
"""
import math

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import SPEED_MARGIN, load_data, fit_model


def main():
    print("正在讀取資料...")
    df = load_data()
    print(f"有效樣本數：{len(df):,} 筆（{df['Time'].dt.date.nunique()} 天、{df['VDID'].nunique()} 支 VD）")
    print(f"平均車速：內側 {df['Speed_Inner'].mean():.1f} km/h、外側 {df['Speed_Outer'].mean():.1f} km/h")
    print(f"平均佔有率：內側 {df['Occ_Inner'].mean():.1f}%、外側 {df['Occ_Outer'].mean():.1f}%")

    counts = df['Target'].value_counts().sort_index()
    print(f"\n標籤分布（1 = 內側快 {SPEED_MARGIN} km/h 以上）：")
    print(f"  0 建議走外側：{counts[0]:,} 筆 ({counts[0] / len(df):.1%})")
    print(f"  1 建議走內側：{counts[1]:,} 筆 ({counts[1] / len(df):.1%})")

    print("\n開始訓練模型...")
    model, X_test, y_test = fit_model(df)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())   # 永遠猜多數類別的準確率
    b0 = model.intercept_[0]
    b1, b2 = model.coef_[0]

    print("\n" + "=" * 48)
    print("【模型結果】")
    print(f"測試集準確率：{acc:.2%}（只猜多數類別的基準：{baseline:.2%}）")
    print("-" * 48)
    print(f"截距 β0          : {b0:.4f}")
    print(f"內側佔有率 β1 (x1): {b1:.4f}")
    print(f"外側佔有率 β2 (x2): {b2:.4f}")
    print("-" * 48)
    print(f"z = {b0:.4f} + ({b1:.4f})·x1 + ({b2:.4f})·x2")
    print("P(建議走內側) = 1 / (1 + e^(−z))")
    print(f"決策邊界（P = 0.5）：x2 = {-b0 / b2:.4f} + ({-b1 / b2:.4f})·x1")
    print("=" * 48)

    print("\n混淆矩陣（列：實際 0/1，欄：預測 0/1）：")
    print(confusion_matrix(y_test, y_pred))
    print("\n分類報告：")
    print(classification_report(y_test, y_pred, target_names=['走外側 (0)', '走內側 (1)'], digits=3))

    print("【解讀】")
    print(f"- 兩側都沒車時（x1 = x2 = 0），P(建議走內側) = {1 / (1 + math.exp(-b0)):.1%}")
    print(f"- 內側佔有率每增加 1%，勝算比變為 {math.exp(b1):.3f} 倍；外側每增加 1%，變為 {math.exp(b2):.3f} 倍")
    print("請接著執行 03_visualization.py")


if __name__ == '__main__':
    main()
