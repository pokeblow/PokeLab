import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from pathlib import Path

# 当前文件目录
BASE_DIR = Path(__file__).resolve().parent

# buffer 文件路径
BUFFER_FILE = BASE_DIR / "buffer.tmp"

# 读取内容
def read_buffer():
    if not BUFFER_FILE.exists():
        return None
    with open(BUFFER_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


# 兼容：行首可能有 [timestamp]
# 示例：
# [2026-02-23 20:38:27] Epoch 21/50: train: lr • train = 0.0010, ...
EPOCH_LINE_RE = re.compile(
    r"(?:\[[^\]]+\]\s*)?Epoch\s+(\d+)\s*/\s*(\d+)\s*:\s*(.*)"
)

HEADER_EPOCHS_RE = re.compile(r"epochs\s*=\s*(\d+)")
HEADER_VERSION_RE = re.compile(r"version\s*=\s*([^\s|,]+)")

# 直接抓取所有 "metric • split = value"
# 兼容 float / scientific notation
METRIC_KV_RE = re.compile(
    r"([A-Za-z0-9_./-]+)\s*[•]\s*([A-Za-z0-9_./-]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def _read_lines(log_path: str) -> List[str]:
    with open(log_path, "r", encoding="utf-8") as f:
        return f.readlines()


def _parse_header(lines: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Try to extract declared total epochs and version from any header line.
    summary.log 可能没有 version/epochs header，因此允许 None。
    """
    joined = " ".join(lines[:8])
    m_e = HEADER_EPOCHS_RE.search(joined)
    total_epochs = int(m_e.group(1)) if m_e else None

    m_v = HEADER_VERSION_RE.search(joined)
    version = m_v.group(1) if m_v else None
    return total_epochs, version


def _parse_epochs(lines: List[str]) -> List[int]:
    epochs = set()
    for line in lines:
        m = re.search(r"(?:\[[^\]]+\]\s*)?Epoch\s+(\d+)\s*/\s*\d+\s*:", line)
        if m:
            epochs.add(int(m.group(1)))
    return sorted(epochs)


def _collect_metrics(lines: List[str]):
    """
    Parse training/validation metrics from log lines.

    Returns
    -------
    metrics : dict[metric][split] -> {epoch: value}

    适配你的 log：
    "... lr • train = 0.0010, loss • train = 1.1127, acc • train = 0.3142, | valid: loss • valid = 1.0762, ..."
    直接用正则抓 "metric • split = value" 即可，不依赖 train:/valid: 段落结构。
    """
    metrics = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        m = EPOCH_LINE_RE.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        tail = m.group(3)

        for mm in METRIC_KV_RE.finditer(tail):
            metric_name = mm.group(1).strip()
            split = mm.group(2).strip()
            try:
                v = float(mm.group(3))
            except Exception:
                continue
            metrics[metric_name][split][epoch] = v

    return metrics


def plot_training_log(
    log_path: str = '',
    out_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (9.5, 6.2),
    dpi: int = 180,
    linewidth: float = 2.2,
    markersize: float = 5.5,
    grid_alpha: float = 0.35,
    annotate_last: bool = True,
    save_svg: bool = True,
    show: bool = True,
) -> Dict[str, str]:
    """
    Read a training log file and plot metrics.
    One plot per metric; different splits (train/valid/...) shown in one figure.

    Parameters
    ----------
    log_path : str
        Path to training log file. If empty, use G.get_log_root().
    out_dir : Optional[str]
        Where to save figures. Defaults to "<log_dir>/figs_<log_basename_noext>".
    show : bool
        If True, display figures interactively.

    Returns
    -------
    Dict[str, str]
        Mapping from metric name to saved PNG path.
    """
    if not log_path:
        log_path = f'{read_buffer()}/summary.log'

    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    lines = _read_lines(log_path)
    epochs = _parse_epochs(lines)
    if not epochs:
        raise ValueError("No 'Epoch k/N' lines found in log.")

    total_epochs_header, version = _parse_header(lines)
    metrics = _collect_metrics(lines)
    if not metrics:
        raise ValueError("No 'metric • split = value' patterns found in log.")

    # out_dir default
    log_dir = os.path.dirname(os.path.abspath(log_path))
    base = os.path.splitext(os.path.basename(log_path))[0]
    if out_dir is None:
        out_dir = os.path.join(log_dir, f"figs_{base}")
    os.makedirs(out_dir, exist_ok=True)

    title_suffix_parts = []
    if version:
        title_suffix_parts.append(f"version={version}")
    if total_epochs_header:
        title_suffix_parts.append(f"epochs={total_epochs_header}")
    title_suffix = "  ·  ".join(title_suffix_parts) if title_suffix_parts else ""

    saved: Dict[str, str] = {}

    for metric_name, splits in metrics.items():
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # draw each split
        # 优先把 train/valid 排前面（如果存在）
        split_order = ["train", "valid"]
        split_names = sorted(
            splits.keys(),
            key=lambda s: (split_order.index(s) if s in split_order else 999, s),
        )

        for split_name in split_names:
            epoch_map = splits[split_name]
            xs, ys = [], []
            for e in epochs:
                if e in epoch_map:
                    xs.append(e)
                    ys.append(epoch_map[e])

            if not xs:
                continue

            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=linewidth,
                markersize=markersize,
                label=split_name,
            )

            if annotate_last:
                ax.annotate(
                    f"{ys[-1]:.4f}",
                    (xs[-1], ys[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=10,
                )

        # axis & grid
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)

        title = f"{metric_name} over epochs"
        if title_suffix:
            title = f"{title}\n{title_suffix}"
        ax.set_title(title, fontsize=14, pad=10)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))

        ax.grid(True, which="major", linestyle="--", alpha=grid_alpha)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=grid_alpha * 0.8)

        # epoch range
        ax.set_xlim(min(epochs), max(epochs))

        leg = ax.legend(title="split", frameon=True, framealpha=0.9)
        if leg and leg.get_title():
            leg.get_title().set_fontsize(10)

        fig.tight_layout()

        # save
        png_path = os.path.join(out_dir, f"{base}__{metric_name}.png")
        fig.savefig(png_path, dpi=dpi)
        saved[metric_name] = png_path

        if save_svg:
            svg_path = os.path.join(out_dir, f"{base}__{metric_name}.svg")
            fig.savefig(svg_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved

if __name__ == "__main__":
    plot_training_log()
