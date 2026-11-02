"""
產生兩張圖到 figures/：
  lane_speed_by_hour.png  各時段內、外側車道的平均車速
  decision_boundary.png   模型的決策邊界與 P(建議走內側) 機率分布
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import FEATURES, FIG_DIR, load_data, fit_model

# 配色：內側 = 藍、外側 = 橙（同一個車道在兩張圖用同一個顏色）
INNER_COLOR = '#2a78d6'
OUTER_COLOR = '#eb6834'
NEUTRAL = '#f0efec'
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'text.color': TEXT,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
})


def plot_speed_by_hour(df):
    hourly = df.groupby(df['Time'].dt.hour)[['Speed_Inner', 'Speed_Outer']].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    for col, color, label in [('Speed_Inner', INNER_COLOR, 'Inner lane (LaneID 1)'),
                              ('Speed_Outer', OUTER_COLOR, 'Outer lane (LaneID 0)')]:
        ax.plot(hourly.index, hourly[col], color=color, linewidth=2, marker='o',
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f'{hourly[col].iloc[-1]:.0f} km/h', (hourly.index[-1], hourly[col].iloc[-1]),
                    xytext=(8, 0), textcoords='offset points', va='center', color=TEXT_2)

    ax.set_title('Mean speed by hour, Hsuehshan Tunnel (weekends)', loc='left')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Mean speed (km/h)')
    ax.set_xticks(hourly.index)
    ax.set_xlim(hourly.index.min() - 0.5, hourly.index.max() + 1.8)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'lane_speed_by_hour.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_decision_boundary(df, model, acc):
    b0 = model.intercept_[0]
    b1, b2 = model.coef_[0]

    fig, ax = plt.subplots(figsize=(8, 7))

    # 背景：P(建議走內側) 的機率分布（橙 = 外側、灰 = 0.5、藍 = 內側）
    grid = np.linspace(0, 50, 201)
    g1, g2 = np.meshgrid(grid, grid)
    grid_df = pd.DataFrame({FEATURES[0]: g1.ravel(), FEATURES[1]: g2.ravel()})
    prob = model.predict_proba(grid_df)[:, 1].reshape(g1.shape)
    cmap = LinearSegmentedColormap.from_list('outer_inner', [OUTER_COLOR, NEUTRAL, INNER_COLOR])
    cf = ax.contourf(g1, g2, prob, levels=np.linspace(0, 1, 11), cmap=cmap, alpha=0.35)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('P(take the inner lane)')
    cbar.outline.set_visible(False)

    # 散佈點：隨機抽 3000 筆；佔有率是整數，加 ±0.3 的抖動避免點重疊
    sample = df.sample(n=3000, random_state=42)
    rng = np.random.default_rng(0)
    for target, color, label in [(1, INNER_COLOR, 'Inner lane faster by > 3 km/h'),
                                 (0, OUTER_COLOR, 'Otherwise (take outer lane)')]:
        s = sample[sample['Target'] == target]
        jitter = rng.uniform(-0.3, 0.3, size=(len(s), 2))
        ax.scatter(s[FEATURES[0]] + jitter[:, 0], s[FEATURES[1]] + jitter[:, 1], s=14,
                   color=color, alpha=0.6, linewidths=0, label=label)

    # 決策邊界 P = 0.5：β0 + β1·x1 + β2·x2 = 0
    x1 = np.array([0, 50])
    ax.plot(x1, -(b0 + b1 * x1) / b2, '--', color=TEXT, linewidth=2,
            label=f'Decision boundary: x2 = {-b0 / b2:.2f} + {-b1 / b2:.2f}·x1')

    ax.set_title(f'Lane choice model (test accuracy {acc:.1%})', loc='left')
    ax.set_xlabel('Inner-lane occupancy x1 (%)')
    ax.set_ylabel('Outer-lane occupancy x2 (%)')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, loc='upper left', fontsize=9.5)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'decision_boundary.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_data()
    model, X_test, y_test = fit_model(df)
    acc = model.score(X_test, y_test)

    print('已儲存：', plot_speed_by_hour(df))
    print('已儲存：', plot_decision_boundary(df, model, acc))


if __name__ == '__main__':
    main()
