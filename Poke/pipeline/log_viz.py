
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from . import globals as G

__all__ = ["plot_training_log"]

EPOCH_LINE_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)\s*:\s*(.*)")
HEADER_EPOCHS_RE = re.compile(r"epochs\s*=\s*(\d+)")

def _read_lines(log_path: str) -> List[str]:
    with open(log_path, "r", encoding="utf-8") as f:
        return f.readlines()

def _parse_header(lines: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Try to extract declared total epochs and version from any header line."""
    joined = " ".join(lines[:5])
    m_e = HEADER_EPOCHS_RE.search(joined)
    total_epochs = int(m_e.group(1)) if m_e else None
    m_v = re.search(r"version\s*=\s*([^\s|,]+)", joined)
    version = m_v.group(1) if m_v else None
    return total_epochs, version

def _parse_epochs(lines: List[str]) -> List[int]:
    epochs = set()
    for line in lines:
        m = re.search(r"Epoch\s+(\d+)\s*/\s*\d+\s*:", line)
        if m:
            epochs.add(int(m.group(1)))
    return sorted(epochs)

def _collect_metrics(lines: List[str]):
    """
    Parse training/validation metrics from log lines.
    Returns: dict[metric][split] -> {epoch: value}
    """
    metrics = defaultdict(lambda: defaultdict(dict))
    for line in lines:
        m = EPOCH_LINE_RE.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        tail = m.group(3)
        parts = [p.strip() for p in tail.split(",") if p.strip()]
        for part in parts:
            kv = part.split("=")
            if len(kv) != 2:
                continue
            left, val = kv[0].strip(), kv[1].strip()
            m2 = re.match(r"([A-Za-z0-9_./-]+)\s*[•]\s*([A-Za-z0-9_./-]+)", left)
            if m2:
                metric_name = m2.group(1).strip()
                split = m2.group(2).strip()
            else:
                metric_name = left.strip()
                split = "all"
            # numeric conversion with cleanup
            try:
                v = float(val)
            except ValueError:
                val_clean = re.sub(r"[^\d.eE+-]+", "", val)
                try:
                    v = float(val_clean)
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
) -> Dict[str, str]:
    """
    Read a training log file and plot metrics (beautified).
    One plot per metric; train/valid/all with the same metric go on the same figure.

    Parameters
    ----------
    log_path : str
        Path to training log.
    out_dir : Optional[str]
        Where to save figures. Defaults to "<dir>/figs_<basename>".
    figsize : (w, h)
        Figure size in inches.
    dpi : int
        Save resolution.
    linewidth : float
        Line width for metric curves.
    markersize : float
        Marker size.
    grid_alpha : float
        Alpha for major/minor gridlines.
    annotate_last : bool
        If True, annotate latest value beside each line.
    save_svg : bool
        If True, save an additional SVG per figure.

    Returns
    -------
    Dict[str, str]
        Mapping from metric name to PNG path.
    """
    if log_path == '':
        print(G.get_log_root())
        log_path = G.get_log_root()

    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    lines = _read_lines(log_path)
    epochs = _parse_epochs(lines)
    if not epochs:
        raise ValueError("No 'Epoch k/N' lines found in log.")
    total_epochs_header, version = _parse_header(lines)
    metrics = _collect_metrics(lines)

    title_suffix = []
    if version:
        title_suffix.append(f"version={version}")
    if total_epochs_header:
        title_suffix.append(f"epochs={total_epochs_header}")
    title_suffix = "  ·  ".join(title_suffix) if title_suffix else ""

    saved = {}
    for metric_name, splits in metrics.items():
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # draw each split
        for split_name, epoch_map in sorted(splits.items()):
            xs = []
            ys = []
            for e in epochs:
                if e in epoch_map:
                    xs.append(e)
                    ys.append(epoch_map[e])
            if xs:
                line, = ax.plot(xs, ys, marker="o", linewidth=linewidth, markersize=markersize, label=split_name)
                if annotate_last:
                    ax.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]),
                                xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=10)

        # axis & grid prettify
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        title = f"{metric_name} over epochs"
        if title_suffix:
            title = f"{title}\n{title_suffix}"
        ax.set_title(title, fontsize=14, pad=10)

        # integer epoch ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # y tick format to sensible precision
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4g"))

        # major & minor grid
        ax.grid(True, which="major", linestyle="--", alpha=grid_alpha)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=grid_alpha * 0.8)

        # legend inside, best location, nice frame
        leg = ax.legend(title="split", frameon=True, framealpha=0.9)
        if leg and leg.get_title():
            leg.get_title().set_fontsize(10)

        fig.tight_layout()

        # save
        plt.show()
        plt.close(fig)

