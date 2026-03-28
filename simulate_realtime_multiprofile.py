"""
biosignal_generator.py
======================
Genera continuamente biosegnali realistici e li scrive su file CSV in tempo reale.

Controlli:
  Q  →  esce pulitamente

Output: biosignals_live.csv  (una riga al secondo)

Colonne:
  bpm, hrv_rmssd_ms, eda_us, resp_rpm, skin_temp_c, state, event

  state:   0=normal  1=onset  2=panic  3=recovery
  event:   0=none  4=running  5=fright  6=injury  7=sleep
  subject: 0=sedentary  1=athletic  2=anxious  3=elderly  4=relaxed

Profili soggetto (visibili solo a terminale):
  sedentary | athletic | anxious | elderly | relaxed

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

EVENT_PROBABILITY_PER_SEC = 0.1   # ← modifica qui

# Probabilità che parta un attacco di panico ogni secondo (solo in stato normal
# e senza eventi attivi).  Es. 0.002 → circa 1 panico ogni ~500 secondi.
PANIC_PROBABILITY_PER_SEC = 0.2   # ← modifica qui

# Pesi relativi: running, fright, injury, sleep
EVENT_WEIGHTS = [4, 2, 1, 3]      # ← modifica qui

# Durata variabile: ogni volta che parte un evento viene estratto
# un valore casuale uniforme tra min e max (in secondi).
EVENT_DURATION_RANGE = {
    "running": {"active": (10, 30), "recovery": (10, 20)},
    "fright":  {"active": (10, 30), "recovery": (10, 20)},
    "injury":  {"active": (10, 30), "recovery": (10, 20)},
    "sleep":   {"active": (10, 30), "recovery": (10, 20)},
}


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE SOGGETTI  ←  modifica qui
# ════════════════════════════════════════════════════════════════════════════

SUBJECT_CHANGE_EVERY = 5000   # ← ogni quanti campioni cambia soggetto
SUBJECT_RANDOM_ORDER = True   # ← True = ordine casuale, False = ordine fisso

# ── Profili ──────────────────────────────────────────────────────────────────
# "normal"        → parametri (mu, sigma) a riposo del soggetto
# "stress_factor" → moltiplica i delta di stress/eventi (>1 = più reattivo)
# "color"         → colore ANSI per il terminale

SUBJECT_PROFILES = {

    "sedentary": {
        "normal": {
            "bpm":          (75,   5.0),
            "hrv_rmssd_ms": (32,   8.0),
            "eda_us":       (1.5,  0.5),
            "resp_rpm":     (15,   2.0),
            "skin_temp_c":  (34.0, 0.5),
        },
        "stress_factor": 1.15,
        "noise_scale": (0.3, 0.6),
        "color": "\033[33m",   # giallo
    },

    "athletic": {
        "normal": {
            "bpm":          (52,   4.0),
            "hrv_rmssd_ms": (68,  12.0),
            "eda_us":       (0.9,  0.3),
            "resp_rpm":     (11,   1.5),
            "skin_temp_c":  (35.0, 0.4),
        },
        "stress_factor": 0.85,
        "noise_scale": (0.2, 0.4),
        "color": "\033[32m",   # verde
    },

    "anxious": {
        "normal": {
            "bpm":          (80,   7.0),
            "hrv_rmssd_ms": (22,   6.0),
            "eda_us":       (2.8,  0.8),
            "resp_rpm":     (17,   2.5),
            "skin_temp_c":  (33.5, 0.6),
        },
        "stress_factor": 1.40,
        "noise_scale": (0.5, 0.9),
        "color": "\033[31m",   # rosso
    },

    "elderly": {
        "normal": {
            "bpm":          (68,   4.5),
            "hrv_rmssd_ms": (18,   5.0),
            "eda_us":       (0.8,  0.3),
            "resp_rpm":     (14,   1.5),
            "skin_temp_c":  (33.8, 0.5),
        },
        "stress_factor": 1.10,
        "noise_scale": (0.3, 0.5),
        "color": "\033[36m",   # ciano
    },

    "relaxed": {
        "normal": {
            "bpm":          (58,   3.0),
            "hrv_rmssd_ms": (58,  10.0),
            "eda_us":       (0.7,  0.2),
            "resp_rpm":     (10,   1.0),
            "skin_temp_c":  (35.5, 0.3),
        },
        "stress_factor": 0.75,
        "noise_scale": (0.2, 0.4),
        "color": "\033[34m",   # blu
    },
}

PROFILE_NAMES = list(SUBJECT_PROFILES.keys())
PROFILE_CODE  = {name: i for i, name in enumerate(PROFILE_NAMES)}  # sedentary=0 athletic=1 anxious=2 elderly=3 relaxed=4


# ════════════════════════════════════════════════════════════════════════════
#  DELTA DI STRESS/EVENTI (rispetto al normal del soggetto)
# ════════════════════════════════════════════════════════════════════════════
# Questi delta vengono moltiplicati per stress_factor del profilo attivo.

_STATE_DELTA = {
    "onset": {
        "bpm":          +20,
        "hrv_rmssd_ms": -20,
        "eda_us":       +2.3,
        "resp_rpm":      +5,
        "skin_temp_c":  -0.7,
    },
    "panic": {
        "bpm":          +50,
        "hrv_rmssd_ms": -33,
        "eda_us":       +5.3,
        "resp_rpm":     +11,
        "skin_temp_c":  -1.5,
    },
}

_EVENT_DELTA = {
    "running": {
        "bpm":          +80,
        "hrv_rmssd_ms": -27,
        "eda_us":       +3.8,
        "resp_rpm":     +15,
        "skin_temp_c":  +2.0,
    },
    "fright": {
        "bpm":          +35,
        "hrv_rmssd_ms": -25,
        "eda_us":       +3.3,
        "resp_rpm":      +9,
        "skin_temp_c":  -1.0,
    },
    "injury": {
        "bpm":          +30,
        "hrv_rmssd_ms": -23,
        "eda_us":       +2.8,
        "resp_rpm":      +7,
        "skin_temp_c":  -0.7,
    },
    "sleep": {
        "bpm":          -13,
        "hrv_rmssd_ms": +20,
        "eda_us":       -0.7,
        "resp_rpm":      -4,
        "skin_temp_c":  +0.7,
    },
}

_REF_STD = {
    "bpm":           4.0,
    "hrv_rmssd_ms": 10.0,
    "eda_us":         0.4,
    "resp_rpm":       1.5,
    "skin_temp_c":    0.4,
}

CLAMP = {
    "bpm":          (40,  210),
    "hrv_rmssd_ms": (3,   120),
    "eda_us":       (0.2,  30),
    "resp_rpm":     (6,    50),
    "skin_temp_c":  (28,   38),
}

SIGNALS    = list(CLAMP.keys())
EVENT_CODE = {None: 0, "running": 4, "fright": 5, "injury": 6, "sleep": 7}
EVENT_NAMES = list(EVENT_DURATION_RANGE.keys())
DURATION   = {"onset": 35, "panic": 70, "recovery": 80}

SAMPLE_RATE_HZ = float(args.sample_rate)
OUTPUT_FILE    = "biosignals_live.csv"


# ════════════════════════════════════════════════════════════════════════════
#  HELPER: costruisce params (mu, sigma) per un profilo
# ════════════════════════════════════════════════════════════════════════════

def build_params_for_profile(profile_name: str) -> dict:
    profile = SUBJECT_PROFILES[profile_name]
    base    = profile["normal"]
    sf      = profile["stress_factor"]
    result  = {"normal": base}

    for state, delta in _STATE_DELTA.items():
        result[state] = {
            sig: (base[sig][0] + delta[sig] * sf,
                  _REF_STD[sig] * (1 + 0.3 * (sf - 1)))
            for sig in SIGNALS
        }

    for event, delta in _EVENT_DELTA.items():
        result[event] = {
            sig: (base[sig][0] + delta[sig] * sf,
                  _REF_STD[sig] * (1 + 0.3 * (sf - 1)))
            for sig in SIGNALS
        }

    return result


# ════════════════════════════════════════════════════════════════════════════
#  ENGINE
# ════════════════════════════════════════════════════════════════════════════

class BiosignalEngine:

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.state = "normal"

        # Soggetto
        self._profile_idx   = 0
        self._profile_order = PROFILE_NAMES[:]
        if SUBJECT_RANDOM_ORDER:
            self.rng.shuffle(self._profile_order)

        self.current_profile = self._profile_order[0]
        self._params         = build_params_for_profile(self.current_profile)
        self._sample_count   = 0
        self._subject_id     = 0
        self._noise_scale    = self._sample_noise_scale()

        # Evento
        self.event               = None
        self._event_phase        = None
        self._event_phase_t      = 0
        self._event_dur_active   = 0
        self._event_dur_recovery = 0

        # Segnali
        self.current   = {s: self._params["normal"][s][0] for s in SIGNALS}
        self._ar_noise = {s: 0.0 for s in SIGNALS}

        # Panic
        self._phase_t   = 0
        self._lock      = threading.Lock()
        self._panic_req = False

    # ── API ───────────────────────────────────────────────────────────────────

    def step(self) -> tuple:
        with self._lock:
            changed = self._maybe_switch_subject()
            self._update_panic_state()
            self._update_event()
            values = self._sample()
            self._sample_count += 1
        return values, changed

    # ── Cambio soggetto ───────────────────────────────────────────────────────

    def _maybe_switch_subject(self) -> bool:
        if self._sample_count > 0 and self._sample_count % SUBJECT_CHANGE_EVERY == 0:
            self._profile_idx = (self._profile_idx + 1) % len(self._profile_order)
            if self._profile_idx == 0 and SUBJECT_RANDOM_ORDER:
                self.rng.shuffle(self._profile_order)

            self.current_profile = self._profile_order[self._profile_idx]
            self._params         = build_params_for_profile(self.current_profile)
            self._subject_id    += 1
            self._noise_scale    = self._sample_noise_scale()

            # Reset stati (i valori correnti restano → transizione morbida)
            self.state          = "normal"
            self._phase_t       = 0
            self.event               = None
            self._event_phase        = None
            self._event_phase_t      = 0
            self._event_dur_active   = 0
            self._event_dur_recovery = 0
            return True
        return False

    # ── Panic FSM ────────────────────────────────────────────────────────────

    def _update_panic_state(self):
        # Attivazione casuale (solo se normal e nessun evento attivo)
        if (self.state == "normal"
                and self._event_phase is None
                and self.rng.random() < PANIC_PROBABILITY_PER_SEC):
            self.state        = "onset"
            self._phase_t     = 0
            self.event        = None   # interrompe qualsiasi evento in corso
            self._event_phase = None
            self._event_phase_t = 0
            print("\n  \u26a1  Attacco di panico casuale!\n", flush=True)
            return

        self._phase_t += 1
        transitions = {
            "onset":    ("panic",    DURATION["onset"]),
            "panic":    ("recovery", DURATION["panic"]),
            "recovery": ("normal",   DURATION["recovery"]),
        }
        if self.state in transitions:
            nxt, dur = transitions[self.state]
            if self._phase_t >= dur:
                self.state    = nxt
                self._phase_t = 0

    # ── Event FSM ────────────────────────────────────────────────────────────

    def _update_event(self):
        if self.state != "normal":
            return

        if self._event_phase is not None:
            self._event_phase_t += 1
            dur = (self._event_dur_active
                   if self._event_phase == "active"
                   else self._event_dur_recovery)
            if self._event_phase_t >= dur:
                if self._event_phase == "active":
                    self._event_phase   = "recovery"
                    self._event_phase_t = 0
                else:
                    self.event        = None
                    self._event_phase = None
            return

        if self.rng.random() < EVENT_PROBABILITY_PER_SEC:
            w = np.array(EVENT_WEIGHTS, dtype=float)
            self.event              = self.rng.choice(EVENT_NAMES, p=w / w.sum())
            self._event_phase       = "active"
            self._event_phase_t     = 0
            # Campiona durate casuali per questa istanza dell'evento
            r = EVENT_DURATION_RANGE[self.event]
            self._event_dur_active   = int(self.rng.integers(r["active"][0],   r["active"][1]   + 1))
            self._event_dur_recovery = int(self.rng.integers(r["recovery"][0], r["recovery"][1] + 1))

    # ── Campionamento ─────────────────────────────────────────────────────────

    def _sample_noise_scale(self) -> float:
        """Campiona un noise_scale casuale dal range del profilo corrente."""
        lo, hi = SUBJECT_PROFILES[self.current_profile]["noise_scale"]
        return float(self.rng.uniform(lo, hi))

    def _sample(self) -> dict:
        params = self._resolve_params()
        alpha  = self._blend_alpha()
        out    = {}

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
        if self.state == "onset":    return self._params["onset"]
        if self.state == "panic":    return self._params["panic"]
        if self.state == "recovery": return self._params["normal"]
        if self._event_phase == "active" and self.event in self._params:
            return self._params[self.event]
        return self._params["normal"]

    def _blend_alpha(self) -> float:
        if self.state == "onset":    return 0.08
        if self.state == "panic":    return 0.10
        if self.state == "recovery": return 0.05
        if self._event_phase == "active":
            return {"running": 0.07, "fright": 0.20, "injury": 0.12, "sleep": 0.03}.get(self.event, 0.07)
        if self._event_phase == "recovery": return 0.04
        return 0.03


# ════════════════════════════════════════════════════════════════════════════
#  CSV
# ════════════════════════════════════════════════════════════════════════════

def open_csv(path: str):
    p      = Path(path)
    is_new = not p.exists() or p.stat().st_size == 0
    fobj   = open(p, "a", newline="", buffering=1)
    fields = SIGNALS + ["state", "event", "subject"]
    writer = csv.DictWriter(fobj, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return fobj, writer


# ════════════════════════════════════════════════════════════════════════════
#  TASTIERA
# ════════════════════════════════════════════════════════════════════════════

class KeyHandler:
    def __init__(self, engine: BiosignalEngine, stop_event: threading.Event):
        self.engine, self.stop_event = engine, stop_event

    def on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, "char") else None
        except AttributeError:
            ch = None
        if ch == "q":
            self.stop_event.set()
            return False


# ════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ════════════════════════════════════════════════════════════════════════════

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
PHASE_LABEL = {None: "", "active": "▶", "recovery": "↩"}


def print_subject_banner(subject_id: int, profile_name: str):
    col   = SUBJECT_PROFILES[profile_name]["color"]
    reset = "\033[0m"
    p     = SUBJECT_PROFILES[profile_name]["normal"]
    sf    = SUBJECT_PROFILES[profile_name]["stress_factor"]
    print()
    print(f"  {'─'*58}")
    print(f"  {col}▶▶  NUOVO SOGGETTO #{subject_id:03d}  —  profilo: {profile_name.upper()}{reset}")
    print(f"  {'─'*58}")
    ns = SUBJECT_PROFILES[profile_name]["noise_scale"]
    print(f"     BPM:{p['bpm'][0]}  HRV:{p['hrv_rmssd_ms'][0]}ms  "
          f"EDA:{p['eda_us'][0]}µS  Resp:{p['resp_rpm'][0]}rpm  "
          f"Temp:{p['skin_temp_c'][0]}°C  stress_factor:{sf}  noise:{ns}")
    print(f"  {'─'*58}")
    print()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Biosignal Generator  —  avviato")
    print("=" * 62)
    print(f"  Output         : {OUTPUT_FILE}")
    print(f"  Sample rate    : {SAMPLE_RATE_HZ} Hz")
    print(f"  Event prob     : {EVENT_PROBABILITY_PER_SEC*100:.2f}% / sec")
    print(f"  Panic prob     : {PANIC_PROBABILITY_PER_SEC*100:.3f}% / sec")
    print(f"  Subject change : ogni {SUBJECT_CHANGE_EVERY} campioni")
    print(f"  Profili        : {', '.join(PROFILE_NAMES)}")
    print()
    print("  Q  →  esci")
    print("=" * 62)

    engine     = BiosignalEngine(seed=None)
    stop_event = threading.Event()
    handler    = KeyHandler(engine, stop_event)

    print_subject_banner(engine._subject_id, engine.current_profile)

    listener = keyboard.Listener(on_press=handler.on_press)
    listener.start()

    fobj, writer = open_csv(OUTPUT_FILE)
    t_s = 0

    try:
        while not stop_event.is_set():# and t_s < SUBJECT_CHANGE_EVERY*5:
            tick_start = time.perf_counter()

            values, changed = engine.step()

            if changed:
                print_subject_banner(engine._subject_id, engine.current_profile)

            state_code = {"normal": 0, "onset": 1, "panic": 2, "recovery": 3}.get(engine.state, 0)
            event_code = EVENT_CODE.get(engine.event, 0)

            subject_code = PROFILE_CODE.get(engine.current_profile, 0)
            writer.writerow({**values, "state": state_code, "event": event_code, "subject": subject_code})

            col   = SUBJECT_PROFILES[engine.current_profile]["color"]
            reset = "\033[0m"
            print(
                f"  t={t_s:>5}s  {col}{engine.current_profile:<10}{reset}  "
                f"state:{STATE_LABEL.get(engine.state, engine.state)}  "
                f"event:{EVENT_LABEL.get(engine.event, '         ')}"
                f"{PHASE_LABEL.get(engine._event_phase, '')}  "
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
