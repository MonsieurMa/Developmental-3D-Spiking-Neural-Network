from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def setup_chinese_font(verbose: bool = False) -> str | None:
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"中文字体: {candidate}")
            return candidate
    plt.rcParams["axes.unicode_minus"] = False
    return None


def visualize_brain(brain, save_path: str | Path | None = None):
    setup_chinese_font()
    alive = [neuron for neuron in brain.neurons if neuron.alive]
    fig = plt.figure(figsize=(12, 6), dpi=110)
    ax = fig.add_subplot(121, projection="3d")
    codes = list(brain.regions)
    palette = plt.cm.tab20(np.linspace(0, 1, len(codes)))
    colors = {code: palette[i] for i, code in enumerate(codes)}
    positions = np.array([neuron.position for neuron in alive])
    point_colors = [colors.get(neuron.region, "gray") for neuron in alive]
    sizes = [8.0 + 22.0 * min(3.0, neuron.recent_rate(brain.time, 100.0)) for neuron in alive]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=point_colors, s=sizes, alpha=0.78, depthshade=False)
    ax.set_title("BionicBrain 3D 发育状态")
    ax.set_xlabel("x 左右"); ax.set_ylabel("y 前后"); ax.set_zlabel("z 上下")
    ax2 = fig.add_subplot(122)
    rates = brain.get_region_activity()
    active = sorted(((code, rate) for code, rate in rates.items() if rate > 0), key=lambda item: item[1], reverse=True)[:20]
    if active:
        ax2.barh([item[0] for item in active][::-1], [item[1] for item in active][::-1], color="#4f86f7")
    else:
        ax2.text(0.5, 0.5, "暂无放电记录", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("最近 100 ms 区域放电率")
    ax2.set_xlabel("Hz")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"图已保存: {save_path}")
        plt.close(fig)
    else:
        try:
            plt.show()
        finally:
            plt.close(fig)
    return fig
