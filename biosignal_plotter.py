"""
biosignal_plotter.py
====================
Legge biosignals_live.csv in tempo reale — grafici stile ECG con scala dinamica.

Uso:
  python biosignal_plotter.py
  python biosignal_plotter.py --file mio_file.csv --window 90

Colonne CSV attese:
  bpm, hrv_rmssd_ms, eda_us, resp_rpm, skin_temp_c, state, event

  state:  0=normal  1=onset  2=panic  3=recovery
  event:  0=none  4=running  5=fright  6=injury  7=sleep

Dipendenze:
  pip install matplotlib pandas numpy
"""

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Argomenti ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file",   default="biosignals_live.csv")
parser.add_argument("--window", type=int, default=60)
args = parser.parse_args()

CSV_PATH    = Path(args.file)
WINDOW_SIZE = args.window

# ── Canali ────────────────────────────────────────────────────────────────────
CHANNELS = [
    {"col": "bpm",          "label": "BPM",          "unit": "bpm", "color": "#ff4f4f"},
    {"col": "hrv_rmssd_ms", "label": "HRV (RMSSD)",  "unit": "ms",  "color": "#4fa3ff"},
    {"col": "eda_us",       "label": "EDA",           "unit": "µS",  "color": "#ffb347"},
    {"col": "resp_rpm",     "label": "Respirazione",  "unit": "rpm", "color": "#4fcf8a"},
    {"col": "skin_temp_c",  "label": "Temp. cutanea", "unit": "°C",  "color": "#c77dff"},
]
N_CH = len(CHANNELS)

# ── Metadati stati panico ─────────────────────────────────────────────────────
STATE_META = {
    0: {"label": "NORMAL",   "color": "#2ecc71", "bg": "#0a0a0a", "band": "#2ecc71", "alpha": 0.00},
    1: {"label": "ONSET",    "color": "#f39c12", "bg": "#140e00", "band": "#f39c12", "alpha": 0.13},
    2: {"label": "PANIC",    "color": "#e74c3c", "bg": "#150303", "band": "#e74c3c", "alpha": 0.20},
    3: {"label": "RECOVERY", "color": "#3498db", "bg": "#030d15", "band": "#3498db", "alpha": 0.12},
}

# ── Metadati eventi life ──────────────────────────────────────────────────────
EVENT_META = {
    0: {"label": "",         "color": "#ffffff", "band": "#ffffff", "alpha": 0.00},
    4: {"label": "RUNNING",  "color": "#00e5ff", "band": "#00e5ff", "alpha": 0.10},
    5: {"label": "FRIGHT",   "color": "#ffd600", "band": "#ffd600", "alpha": 0.10},
    6: {"label": "INJURY",   "color": "#ff6d00", "band": "#ff6d00", "alpha": 0.10},
    7: {"label": "SLEEP",    "color": "#b388ff", "band": "#b388ff", "alpha": 0.10},
}

# ── Buffers ───────────────────────────────────────────────────────────────────
buffers   = {ch["col"]: deque(maxlen=WINDOW_SIZE) for ch in CHANNELS}
buf_state = deque(maxlen=WINDOW_SIZE)
buf_event = deque(maxlen=WINDOW_SIZE)
last_row  = 0
total_rows = 0   # contatore assoluto per asse X

def read_new_rows():
    global last_row, total_rows
    if not CSV_PATH.exists():
        return
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return
    new = df.iloc[last_row:]
    if new.empty:
        return
    for _, row in new.iterrows():
        buf_state.append(int(float(row.get("state", 0))))
        buf_event.append(int(float(row.get("event", 0))))
        for ch in CHANNELS:
            col = ch["col"]
            if col in row:
                buffers[col].append(float(row[col]))
        total_rows += 1
    last_row = len(df)

# ── Figura ────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 11), facecolor="#0a0a0a")
fig.canvas.manager.set_window_title("Biosignal Monitor")

outer = gridspec.GridSpec(
    1, 2, width_ratios=[1, 9],
    left=0.01, right=0.98, top=0.91, bottom=0.06, wspace=0.01
)
gs_charts = gridspec.GridSpecFromSubplotSpec(
    N_CH, 1, subplot_spec=outer[1], hspace=0.06
)

axes       = []
lines      = []
fill_polys = []
val_texts  = []
label_axes = []

for i, ch in enumerate(CHANNELS):
    # ── Pannello etichetta sinistro ───────────────────────────────────────────
    lax = fig.add_subplot(
        gridspec.GridSpecFromSubplotSpec(N_CH, 1, subplot_spec=outer[0], hspace=0.06)[i]
    )
    lax.set_facecolor("#111111")
    lax.set_xticks([]); lax.set_yticks([])
    for sp in lax.spines.values():
        sp.set_visible(False)
    lax.spines["right"].set_visible(True)
    lax.spines["right"].set_color("#222222")

    lax.text(0.5, 0.72, ch["label"],
             transform=lax.transAxes, ha="center", va="center",
             fontsize=8, color="#666666", fontweight="bold", fontfamily="monospace")
    vt = lax.text(0.5, 0.30, "—",
                  transform=lax.transAxes, ha="center", va="center",
                  fontsize=18, color=ch["color"], fontweight="bold", fontfamily="monospace")
    lax.text(0.5, 0.08, ch["unit"],
             transform=lax.transAxes, ha="center", va="center",
             fontsize=7, color="#444444", fontfamily="monospace")

    label_axes.append(lax)
    val_texts.append(vt)

    # ── Pannello grafico destro ───────────────────────────────────────────────
    ax = fig.add_subplot(gs_charts[i])
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(0, WINDOW_SIZE)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#1c1c1c", linewidth=0.7, linestyle="-")
    ax.grid(axis="x", color="#161616", linewidth=0.5, linestyle=":")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#222222")
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=7.5, colors="#444444", length=3, pad=4)
    ax.tick_params(axis="x",
                   bottom=(i == N_CH - 1), labelbottom=(i == N_CH - 1),
                   labelsize=7.5, colors="#444444", length=3)

    poly = ax.fill_between([], [], alpha=0)
    fill_polys.append(poly)
    ln, = ax.plot([], [], color=ch["color"], linewidth=1.3, antialiased=True, zorder=3)
    lines.append(ln)
    axes.append(ax)

axes[-1].set_xlabel("campioni", fontsize=8, color="#444444", labelpad=4)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.965, "BIOSIGNAL MONITOR",
         ha="center", fontsize=12, color="#cccccc",
         fontweight="bold", fontfamily="monospace")

# Badge stato panico (sinistra)
state_badge = fig.text(
    0.30, 0.942, "● NORMAL",
    ha="center", fontsize=9, color=STATE_META[0]["color"],
    fontweight="bold", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1f15",
              edgecolor=STATE_META[0]["color"], linewidth=1.2)
)

# Badge evento life (destra)
event_badge = fig.text(
    0.70, 0.942, "",
    ha="center", fontsize=9, color="#ffffff",
    fontweight="bold", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0a",
              edgecolor="#333333", linewidth=1.2)
)

# Contatore campioni (estrema destra)
time_text = fig.text(0.97, 0.965, "t = 0",
                     ha="right", fontsize=8, color="#444444", fontfamily="monospace")

# Legenda eventi (in alto al centro)
legend_items = [(m["color"], m["label"]) for k, m in EVENT_META.items() if m["label"]]
lx = 0.38
for col, lbl in legend_items:
    fig.text(lx, 0.930, "■", color=col, fontsize=9, fontfamily="monospace", alpha=0.8)
    fig.text(lx + 0.018, 0.930, lbl, color="#666666", fontsize=7,
             fontfamily="monospace", va="center")
    lx += 0.072


# ── Helpers ───────────────────────────────────────────────────────────────────

def dynamic_ylim(data, margin_pct=0.30):
    if len(data) < 2:
        return data[0] - 1, data[0] + 1
    lo, hi = data.min(), data.max()
    span   = max(hi - lo, 0.5)
    m      = span * margin_pct
    return lo - m, hi + m


def paint_bands(ax, states, events, n):
    """Disegna bande colorate per stati panico ed eventi life."""
    for patch in list(ax.patches):
        patch.remove()

    # Prima gli eventi (sotto), poi lo stato panico (sopra, più visibile)
    for buf, meta_dict in [(events, EVENT_META), (states, STATE_META)]:
        if len(buf) != n:
            continue
        j = 0
        while j < n:
            val = buf[j]; start = j
            while j < n and buf[j] == val:
                j += 1
            m = meta_dict.get(val, {})
            alpha = m.get("alpha", 0)
            if alpha > 0:
                ax.axvspan(start, j, color=m["band"], alpha=alpha,
                           linewidth=0, zorder=0)


# ── Update ────────────────────────────────────────────────────────────────────

def update(_frame):
    read_new_rows()
    if not buf_state:
        return lines

    cur_state = buf_state[-1]
    cur_event = buf_event[-1]
    s_meta    = STATE_META.get(cur_state, STATE_META[0])
    e_meta    = EVENT_META.get(cur_event, EVENT_META[0])

    # Badge stato
    state_badge.set_text(f"● {s_meta['label']}")
    state_badge.set_color(s_meta["color"])
    state_badge.get_bbox_patch().set_facecolor(s_meta["bg"])
    state_badge.get_bbox_patch().set_edgecolor(s_meta["color"])

    # Badge evento (visibile solo se attivo)
    if cur_event != 0:
        event_badge.set_text(f"▶ {e_meta['label']}")
        event_badge.set_color(e_meta["color"])
        event_badge.get_bbox_patch().set_edgecolor(e_meta["color"])
        event_badge.get_bbox_patch().set_facecolor("#0a0a0a")
    else:
        event_badge.set_text("")
        event_badge.get_bbox_patch().set_edgecolor("#333333")

    time_text.set_text(f"t = {total_rows}")
    fig.set_facecolor(s_meta["bg"])

    states = list(buf_state)
    events = list(buf_event)

    for i, ch in enumerate(CHANNELS):
        col  = ch["col"]
        data = np.array(buffers[col])
        if len(data) == 0:
            continue

        n      = len(data)
        xd     = np.arange(n)
        lo, hi = dynamic_ylim(data)

        axes[i].set_ylim(lo, hi)
        axes[i].set_xlim(0, max(WINDOW_SIZE, n))
        mid = (lo + hi) / 2
        axes[i].set_yticks([lo, mid, hi])
        axes[i].set_yticklabels([f"{lo:.1f}", f"{mid:.1f}", f"{hi:.1f}"])

        lines[i].set_data(xd, data)

        fill_polys[i].remove()
        fill_polys[i] = axes[i].fill_between(
            xd, data, lo, color=ch["color"], alpha=0.10, zorder=1
        )

        paint_bands(axes[i], states, events, n)

        # Cursore verticale
        for vl in getattr(axes[i], "_vlines", []):
            try: vl.remove()
            except Exception: pass
        axes[i]._vlines = [
            axes[i].axvline(n - 1, color="#2a2a2a",
                            linewidth=0.8, linestyle="--", zorder=2)
        ]

        # Valore corrente nel pannello laterale
        cur_val = data[-1]
        val_texts[i].set_text(f"{cur_val:.1f}")
        val_texts[i].set_color("#ff3333" if cur_state == 2 else ch["color"])
        label_axes[i].set_facecolor("#1a0404" if cur_state == 2 else "#111111")

    return lines


ani = animation.FuncAnimation(
    fig, update, interval=100, blit=False, cache_frame_data=False
)

plt.show()