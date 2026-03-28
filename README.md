# LIMEN — Bracciale Bionico per la Gestione Predittiva degli Attacchi di Panico

> *"Limen"* (Latino) — *soglia*. Il confine esatto tra la calma e la crisi.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![TFLite](https://img.shields.io/badge/TFLite_Micro-Edge_AI-green?style=flat-square)](https://www.tensorflow.org/lite/microcontrollers)
[![Hardware](https://img.shields.io/badge/MCU-ESP32--S3-red?style=flat-square)](https://www.espressif.com/en/products/socs/esp32-s3)

---

## Il Problema

Gli attacchi di panico colpiscono senza preavviso — ma il corpo lo sa sempre prima. Nei minuti precedenti una crisi, il sistema nervoso invia segnali fisiologici chiari e inequivocabili: l'HRV cala, l'EDA aumenta, il respiro si accelera. Il problema è che nessuno li sta ascoltando.

I wearable esistenti (smartwatch, fitness band) rilevano la tachicardia *durante* l'attacco, quando è ormai troppo tardi per intervenire. **LIMEN è progettato per agire *prima* che la soglia venga superata** — utilizzando il riconoscimento predittivo di pattern biometrici e un feedback aptico adattivo per de-escalare il sistema nervoso in tempo reale, direttamente sul polso.

---

## Cosa Abbiamo Costruito

LIMEN è un **bracciale intelligente** che monitora continuamente 5 segnali biometrici, esegue una rete neurale LSTM direttamente sul dispositivo per predire l'insorgenza di un attacco di panico con circa 60 secondi di anticipo, e risponde con un protocollo aptico multi-fase adattivo progettato per riportare il corpo alla calma.

L'intero pipeline di inferenza gira **localmente sul microcontrollore** — nessuno smartphone, nessun cloud, nessuna latenza.

---

## Architettura di Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                       BRACCIALE LIMEN                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │    SENSORI   │    │        EDGE AI CORE (ESP32-S3)       │   │
│  │              │    │                                      │   │
│  │  PPG  ──────────► │  Acquisizione e Filtraggio Segnali   │   │
│  │  EDA  ──────────► │     (rimozione artefatti IMU)        │   │ 
│  │  TEMP ──────────► │                  │                   │   │
│  │  IMU  ──────────► │  Finestra scorrevole da 60 secondi   │   │
│  │  RESP ──────────► │     (5 feature × 60 campioni)        │   │
│  │              │    │                  │                   │   │
│  └──────────────┘    │    LSTM 2 layer (TFLite Micro)       │   │ 
│                      │    quantizzato INT8 · < 150 KB       │   │
│                      │                  │                   │   │
│                      │    ┌─────────────▼──────────────┐    │   │
│                      │    │   Classificazione di Stato  │   │   │
│                      │    │  0:Normale  1:Onset         │   │   │
│                      │    │  2:Panico   3:Recovery      │   │   │
│                      │    └─────────────┬──────────────┘    │   │
│                      │                  │                   │   │
│                      │   Loop Aptico Adattivo (eval 40s)    │   │
│                      └──────────────────┼───────────────────┘   │
│                                         │                       │
│  ┌──────────────────────────────────────▼──────────────────┐    │
│  │          3× MOTORI LRA  (driver DRV2605L)               │    │
│  │    ●12    ●4    ●8   → pattern spaziali sfasati         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componenti Hardware

| Componente | Modello | Posizione | Funzione |
|---|---|---|---|
| **PPG** | MAX30102 (Verde/IR) | Interno polso, centro | Frequenza cardiaca + HRV (RMSSD) |
| **EDA/GSR** | Piastrine Ag/AgCl | Interno polso, lati PPG | Conduttanza cutanea (sudorazione) |
| **Termometro** | MAX30205 (contatto) | Interno polso, adiacente EDA | Monitoraggio vasocostrizione periferica |
| **IMU** | MPU6050 (6 assi) | PCB | Filtraggio artefatti da movimento + stima respirazione |
| **Motori aptici** | 3× LRA + DRV2605L | Esterno polso (ore 12, 4, 8) | Feedback tattile adattivo |
| **MCU / SoC** | ESP32-S3 (o Nordic nRF5340) | PCB | Inferenza edge + BLE |
| **Batteria** | LiPo 250 mAh curva | Esterno, sopra PCB | Autonomia giornaliera |

**Materiali:** Scocca in Policarbonato/ABS · Rivestimento interno in silicone medicale ipoallergenico o TPU · Cinturino in nylon elastico con chiusura a micro-regolazione. Peso totale: **< 40 g**.

---

## Modello AI — Predizione degli Attacchi di Panico

### Architettura

Una rete LSTM a 2 layer, addestrata per classificare lo stato fisiologico in 4 categorie:

```
Input: (50 timestep × 6 feature) — finestra scorrevole @ 1 Hz
       [BPM, HRV_RMSSD, EDA, RESP_RPM, TEMP_CUTE, ID_SOGGETTO]
       
LSTM(16, return_sequences=True) + BatchNorm + Dropout(0.4)
LSTM(8)                          + BatchNorm + Dropout(0.3)
Dense(16, relu) + L2(0.001)
Dense(4, softmax)

Classi di output:
  0 → Normale
  1 → Onset (ansia in aumento — allerta precoce)
  2 → Panico (crisi in atto)
  3 → Recovery (de-escalation)
```

### Pipeline di Addestramento

I dati di training vengono generati da `simulate_realtime_multiprofile.py`, che simula **5 profili di soggetto** (sedentario, atletico, ansioso, anziano, rilassato), ognuno con baseline fisiologica individuale e fattore di reattività allo stress. Il generatore produce un flusso CSV continuo ed etichettato che include episodi di panico ed eventi confondenti (corsa, spavento, infortunio, sonno) per massimizzare la robustezza su scenari reali.

Scelte chiave di addestramento:

- **Windowing per soggetto** — le sequenze non attraversano mai i confini tra soggetti, evitando data leakage
- **Data augmentation** — iniezione di rumore Gaussiano (σ=0.02) raddoppia il training set
- **Batch Normalization** — stabilizza il flusso del gradiente nello stack LSTM
- **ReduceLROnPlateau** — dimezza il learning rate dopo 5 epoche senza miglioramento della val_loss (LR minimo = 1e-5)
- **EarlyStopping** — ripristina i migliori pesi, patience = 12 epoche

### Deployment su Edge

Il modello viene esportato in ONNX → quantizzato a **INT8 via TFLite Micro**, riducendo il peso a **< 150 KB** — abbastanza compatto da eseguire l'inferenza in pochi millisecondi direttamente nella RAM dell'ESP32-S3.

---

## Loop Aptico Adattivo

Il sistema rivaluta la risposta fisiologica dell'utente ogni **40 secondi**, adattando lo stimolo aptico per prevenire l'assuefazione neurale (adattamento dei corpuscoli di Pacini).

```
Ogni 40s: misura HRV + EDA
│
├── HRV > 50 E EDA stabile?
│     └── CASO A: De-escalation 
│           Il pattern aptico rallenta: 0.3 Hz → 0.15 Hz → fade out
│
└── Ansia persistente? → CASO B: Shift Strategico
      │
      ├── 0–40s    FASE 1 · Entrainment Respiratorio
      │             Sinusoide globale e lenta → sincronizza meccanicamente il respiro
      │
      ├── 40–80s   FASE 2 · Pressione Crescente
      │             Rampa a dente di sega → grounding somatico sul polso
      │
      ├── 80–120s  FASE 3 · Stimolazione Spaziale Alternata
      │             3 LRA si attivano in rotazione → aggira l'adattamento neurale
      │
      └── >120s    FASE 4 · Pattern Interrupt SOS
                    Impulsi netti stile codice Morse → forza un'elaborazione
                    cognitiva, distraendo l'amigdala dal loop di panico
```

---

## Avvio Rapido

### 1. Genera i Dati di Training

```bash
pip install pynput numpy

# Profilo singolo — premi P per innescare un attacco di panico, Q per uscire
python data_generation/simulate_realtime.py --sample_rate 1

# Multi-profilo — genera automaticamente 5 archetipi di soggetto  <-- Consigliati
python data_generation/simulate_realtime_multiprofile.py --sample_rate 1
```

Output: `biosignals_live.csv` — una riga al secondo con le colonne `bpm, hrv_rmssd_ms, eda_us, resp_rpm, skin_temp_c, state, event, subject`.

### 2. Addestra il Modello

```bash
pip install tensorflow scikit-learn pandas matplotlib

# Rinomina i CSV generati in training_set.csv e test_set.csv
python attacchiPanico.py
```

Il training gira per un massimo di 100 epoche con early stopping. L'accuratezza finale viene stampata insieme alle predizioni per classe.

### 3. Simulazione Hardware

```bash
python electrical_sim.py
```

Visualizza il loop aptico adattivo su 25 cicli di valutazione — traiettoria HRV + progressione delle fasi aptiche.

---

## Roadmap

- [ ] Esportazione modello in TFLite Micro (quantizzazione INT8)
- [ ] Integrazione firmware ESP32-S3
- [ ] Validazione su dataset clinico reale
- [ ] App companion BLE (visualizzazione stato in tempo reale)
- [ ] Calibrazione baseline per utente (wizard di configurazione al primo avvio)

---

## Team

Realizzato dal **TEAM 3**, [28/03/2026]

---
