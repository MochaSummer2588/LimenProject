"""
biosignal_generator.py
======================
Genera continuamente biosegnali realistici e li scrive su file CSV in tempo reale.

Controlli:
  P  →  innesca un attacco di panico (onset → panic → recovery → normal)
  Q  →  esce pulitamente

Output: biosignals_live.csv  (una riga al secondo)

Colonne:
  bpm, hrv_rmssd_ms, eda_us, resp_rpm, skin_temp_c, state, event

  state:  0=normal  1=onset  2=panic  3=recovery
  event:  0=none  4=running  5=fright  6=injury  7=sleep

Dipendenze:
  pip install pynput numpy
"""

import csv
import math
import threading
import time
from pathlib import Path

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample_rate', type=str, default='1',
                    help='Frequenza di campionamento (es. 1, 0.5)')
args = parser.parse_args()

from pynput import keyboard


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE EVENTI  ←  modifica qui
# ════════════════════════════════════════════════════════════════════════════

# Probabilità che un evento parta ogni secondo (mentre state == normal e nessun
# evento è attivo).  Es. 0.003 → circa 1 evento ogni ~330 secondi.
EVENT_PROBABILITY_PER_SEC = 0.1   # ← modifica qui

# Pesi relativi con cui vengono scelti gli eventi (non devono sommare a 1).
# Ordine: running, fright, injury, sleep
EVENT_WEIGHTS = [4, 2, 1, 3]        # ← modifica qui

# Durata in secondi di ciascuna fase di ogni evento.
# Ogni evento ha: "active" (picco) e "recovery" (ritorno al normale).
EVENT_DURATION = {
    "running": {
        "active":   50,#120,   # ← durata attività fisica (secondi)
        "recovery":  25,#60,   # ← recupero dopo la corsa
    },
    "fright": {
        "active":    10,   # ← durata spavento (breve per definizione)
        "recovery":  20,#30,   # ← recupero dopo lo spavento
    },
    "injury": {
        "active":    40,   # ← durata dolore acuto
        "recovery":  20, #90,   # ← recupero dopo l'infortunio
    },
    "sleep": {
        "active":   45,#180,   # ← durata sonno / rilassamento profondo
        "recovery":  30,   # ← ritorno graduale alla veglia
    },
}


# ════════════════════════════════════════════════════════════════════════════
#  PARAMETRI FISIOLOGICI
# ════════════════════════════════════════════════════════════════════════════

STATE_PARAMS = {
    "normal": {
        "bpm":          (65,   4.0),
        "hrv_rmssd_ms": (45,  10.0),
        "eda_us":       (1.2,  0.4),
        "resp_rpm":     (13,   1.5),
        "skin_temp_c":  (34.5, 0.4),
    },
    "onset": {
        "bpm":          (85,   7.0),
        "hrv_rmssd_ms": (25,   7.0),
        "eda_us":       (3.5,  1.0),
        "resp_rpm":     (18,   2.5),
        "skin_temp_c":  (33.8, 0.5),
    },
    "panic": {
        "bpm":          (115, 15.0),
        "hrv_rmssd_ms": (12,   4.0),
        "eda_us":       (6.5,  2.0),
        "resp_rpm":     (24,   4.0),
        "skin_temp_c":  (33.0, 0.6),
    },
}

# Parametri fisiologici per ogni evento (fase attiva)
EVENT_PARAMS = {
    "running": {
        "bpm":          (145, 12.0),   # tachicardia da sforzo
        "hrv_rmssd_ms": (18,   5.0),   # HRV ridotta
        "eda_us":       (5.0,  1.5),   # sudorazione elevata
        "resp_rpm":     (28,   4.0),   # respiro accelerato
        "skin_temp_c":  (36.5, 0.5),   # temperatura alta
    },
    "fright": {
        "bpm":          (100, 10.0),   # picco rapido
        "hrv_rmssd_ms": (20,   6.0),
        "eda_us":       (4.5,  1.2),   # risposta galvanica intensa
        "resp_rpm":     (22,   3.0),   # trattenimento poi iperventilazione
        "skin_temp_c":  (33.5, 0.4),   # vasocostrizione periferica
    },
    "injury": {
        "bpm":          (95,   8.0),   # dolore → tachicardia moderata
        "hrv_rmssd_ms": (22,   6.0),
        "eda_us":       (4.0,  1.0),
        "resp_rpm":     (20,   2.5),
        "skin_temp_c":  (33.8, 0.5),
    },
    "sleep": {
        "bpm":          (52,   3.0),   # bradicardia da sonno
        "hrv_rmssd_ms": (65,  12.0),   # HRV alta = sistema parasimpatico
        "eda_us":       (0.5,  0.2),   # EDA bassissima
        "resp_rpm":     (9,    1.0),   # respiro lento
        "skin_temp_c":  (35.2, 0.3),   # lieve aumento periferico
    },
}

CLAMP = {
    "bpm":          (40,  210),
    "hrv_rmssd_ms": (3,   120),
    "eda_us":       (0.2,  30),
    "resp_rpm":     (6,    50),
    "skin_temp_c":  (28,   38),
}

SIGNALS = list(CLAMP.keys())

# Codici numerici degli eventi per il CSV
EVENT_CODE = {
    None:      0,
    "running": 4,
    "fright":  5,
    "injury":  6,
    "sleep":   7,
}

EVENT_NAMES = list(EVENT_DURATION.keys())

# Durate fasi panico (invariate)
DURATION = {
    "onset":    35,
    "panic":    70,
    "recovery": 80,
}

SAMPLE_RATE_HZ = float(args.sample_rate)
OUTPUT_FILE    = "biosignals_live.csv"


# ════════════════════════════════════════════════════════════════════════════
#  ENGINE
# ════════════════════════════════════════════════════════════════════════════

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BiosignalEngine:
    """
    Macchina a stati principale.

    Panic states:  normal → onset → panic → recovery → normal
    Event states:  none   → <event>_active → <event>_recovery → none

    Panico e eventi non si sovrappongono: il panico ha priorità.
    """

    def __init__(self, seed: int = None):
        self.rng   = np.random.default_rng(seed)
        self.state = "normal"          # panic state machine

        # --- evento corrente ---
        self.event          = None     # nome evento attivo (None = nessuno)
        self._event_phase   = None     # "active" | "recovery" | None
        self._event_phase_t = 0        # tick nella fase corrente

        # Valori correnti
        self.current    = {s: STATE_PARAMS["normal"][s][0] for s in SIGNALS}
        self._ar_noise  = {s: 0.0 for s in SIGNALS}

        # Panic state machine
        self._phase_t   = 0
        self._lock      = threading.Lock()
        self._panic_req = False

    # ── API pubblica ─────────────────────────────────────────────────────────

    def trigger_panic(self):
        with self._lock:
            if self.state == "normal":
                self._panic_req = True

    def step(self) -> dict:
        with self._lock:
            self._update_panic_state()
            self._update_event()
            values = self._sample()
        return values

    # ── State machines ────────────────────────────────────────────────────────

    def _update_panic_state(self):
        if self._panic_req:
            self.state      = "onset"
            self._phase_t   = 0
            self._panic_req = False
            # Interrompe eventuale evento in corso
            self.event        = None
            self._event_phase = None
            return

        self._phase_t += 1
        transitions = {
            "onset":    ("panic",    DURATION["onset"]),
            "panic":    ("recovery", DURATION["panic"]),
            "recovery": ("normal",   DURATION["recovery"]),
        }
        if self.state in transitions:
            next_state, duration = transitions[self.state]
            if self._phase_t >= duration:
                self.state    = next_state
                self._phase_t = 0

    def _update_event(self):
        """Gestisce la macchina a stati degli eventi."""
        # Il panico blocca qualsiasi evento
        if self.state != "normal":
            return

        # Evento in corso → avanza
        if self._event_phase is not None:
            self._event_phase_t += 1
            duration = EVENT_DURATION[self.event][self._event_phase]

            if self._event_phase_t >= duration:
                if self._event_phase == "active":
                    # Passa alla fase di recovery
                    self._event_phase   = "recovery"
                    self._event_phase_t = 0
                else:
                    # Fine evento
                    self.event        = None
                    self._event_phase = None
            return

        # Nessun evento attivo → tenta di avviarne uno casualmente
        if self.rng.random() < EVENT_PROBABILITY_PER_SEC:
            chosen = self.rng.choice(EVENT_NAMES, p=self._normalized_weights())
            self.event          = chosen
            self._event_phase   = "active"
            self._event_phase_t = 0

    def _normalized_weights(self):
        w = np.array(EVENT_WEIGHTS, dtype=float)
        return w / w.sum()

    # ── Campionamento ─────────────────────────────────────────────────────────

    def _sample(self) -> dict:
        params = self._resolve_params()
        alpha  = self._blend_alpha()

        out = {}
        for sig in SIGNALS:
            mu, sigma = params[sig]
            lo, hi    = CLAMP[sig]

            phi = 0.65
            self._ar_noise[sig] = (phi * self._ar_noise[sig]
                                   + self.rng.normal(0, sigma * math.sqrt(1 - phi**2)))

            self.current[sig] += alpha * (mu - self.current[sig])
            raw = self.current[sig] + self._ar_noise[sig] * 0.4

            out[sig] = round(float(np.clip(raw, lo, hi)), 3)

        return out

    def _resolve_params(self) -> dict:
        """Sceglie il set di parametri fisiologici attivi."""
        # Panico ha priorità assoluta
        if self.state == "onset":
            return STATE_PARAMS["onset"]
        if self.state == "panic":
            return STATE_PARAMS["panic"]
        if self.state == "recovery":
            return STATE_PARAMS["normal"]   # target = normale

        # Evento attivo (solo se state == "normal")
        if self._event_phase == "active" and self.event in EVENT_PARAMS:
            return EVENT_PARAMS[self.event]

        # Recovery da evento → blend verso normal
        # (usiamo direttamente normal come target, l'alpha bassa fa il resto)
        return STATE_PARAMS["normal"]

    def _blend_alpha(self) -> float:
        """Velocità di convergenza verso il target."""
        # Panico
        panic_speeds = {
            "onset":    0.08,
            "panic":    0.10,
            "recovery": 0.05,
        }
        if self.state in panic_speeds:
            return panic_speeds[self.state]

        # Evento
        if self._event_phase == "active":
            event_speeds = {
                "running": 0.07,
                "fright":  0.20,   # spavento: reazione rapida
                "injury":  0.12,
                "sleep":   0.03,   # sonno: transizione lenta
            }
            return event_speeds.get(self.event, 0.07)

        if self._event_phase == "recovery":
            return 0.04   # ritorno lento alla normalità

        return 0.03        # normal steady-state


# ════════════════════════════════════════════════════════════════════════════
#  CSV
# ════════════════════════════════════════════════════════════════════════════

def open_csv(path: str):
    p      = Path(path)
    is_new = not p.exists() or p.stat().st_size == 0
    fobj   = open(p, "a", newline="", buffering=1)
    fields = SIGNALS + ["state", "event"]
    writer = csv.DictWriter(fobj, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return fobj, writer


# ════════════════════════════════════════════════════════════════════════════
#  TASTIERA
# ════════════════════════════════════════════════════════════════════════════

class KeyHandler:
    def __init__(self, engine: BiosignalEngine, stop_event: threading.Event):
        self.engine     = engine
        self.stop_event = stop_event

    def on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, "char") else None
        except AttributeError:
            ch = None

        if ch == "p":
            self.engine.trigger_panic()
            print("\n  ⚡  Attacco di panico innescato!\n")
        elif ch == "q":
            self.stop_event.set()
            return False


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

# Etichette display per terminale
STATE_LABEL = {
    "normal":   "\033[32mnormal  \033[0m",
    "onset":    "\033[33monset   \033[0m",
    "panic":    "\033[31mpanic   \033[0m",
    "recovery": "\033[36mrecovery\033[0m",
}
EVENT_LABEL = {
    None:      "         ",
    "running": "\033[35mrunning  \033[0m",
    "fright":  "\033[33mfright   \033[0m",
    "injury":  "\033[31minjury   \033[0m",
    "sleep":   "\033[34msleep    \033[0m",
}
PHASE_LABEL = {
    None:       "",
    "active":   "▶",
    "recovery": "↩",
}


def main():
    print("=" * 60)
    print("  Biosignal Generator  —  avviato")
    print("=" * 60)
    print(f"  Output      : {OUTPUT_FILE}")
    print(f"  Sample rate : {SAMPLE_RATE_HZ} Hz")
    print(f"  Event prob  : {EVENT_PROBABILITY_PER_SEC*100:.2f}% / sec")
    print()
    print("  P  →  innesca attacco di panico")
    print("  Q  →  esci")
    print("=" * 60)
    print()

    engine     = BiosignalEngine(seed=None)
    stop_event = threading.Event()
    handler    = KeyHandler(engine, stop_event)

    listener = keyboard.Listener(on_press=handler.on_press)
    listener.start()

    fobj, writer = open_csv(OUTPUT_FILE)

    t_s = 0
    try:
        while not stop_event.is_set():
            tick_start = time.perf_counter()

            values = engine.step()

            state_code = {
                "normal": 0, "onset": 1, "panic": 2, "recovery": 3
            }.get(engine.state, 0)

            event_code = EVENT_CODE.get(engine.event, 0)

            row = {**values, "state": state_code, "event": event_code}
            writer.writerow(row)

            # ── display ──────────────────────────────────────────────────
            ev_lbl    = EVENT_LABEL.get(engine.event, "         ")
            ph_lbl    = PHASE_LABEL.get(engine._event_phase, "")
            st_lbl    = STATE_LABEL.get(engine.state, engine.state)

            print(
                f"  t={t_s:>5}s  state:{st_lbl}  event:{ev_lbl}{ph_lbl}  "
                f"BPM:{values['bpm']:>5}  "
                f"HRV:{values['hrv_rmssd_ms']:>5}ms  "
                f"EDA:{values['eda_us']:>5}µS  "
                f"Resp:{values['resp_rpm']:>4}rpm  "
                f"Temp:{values['skin_temp_c']:>5}°C",
                flush=True,
            )

            t_s    += 1
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, 1.0 / SAMPLE_RATE_HZ - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        fobj.flush()
        fobj.close()
        listener.stop()
        print(f"\n\n  Chiuso. {t_s} campioni salvati in '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()