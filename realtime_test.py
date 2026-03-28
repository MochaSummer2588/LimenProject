import time
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
print("--- PROTOCOLLO SINTESI: Avvio Modulo di Analisi in Tempo Reale ---")

LOOK_BACK = 50
NOME_MODELLO = 'modello_sintesi_genesi.keras'
NOME_CSV_LIVE = 'biosignals_live.csv'
colonne_input = [0, 1, 2, 3, 4, 7]  # bpm, hrv, eda, resp, temp, subject
nomi_stati = ["Normale", "Si sta alzando", "Panico", "Si sta abbassando"]

POLL_INTERVAL = 0.1

# --- SMOOTHING ---
SMOOTHING_WINDOW = 8
prob_history = deque(maxlen=SMOOTHING_WINDOW)

# --- TRANSIZIONI SEQUENZIALI ---
# L'ordine clinico è: Normale(0) -> Alzando(1) -> Panico(2) -> Abbassando(3) -> Normale(0)
# Non si può saltare: per arrivare a Panico DEVI passare per Alzando.
# Per tornare a Normale DEVI passare per Abbassando.
#
# Per SALIRE di un gradino servono CONFERMA_SALITA predizioni consecutive.
# Per SCENDERE di un gradino servono CONFERMA_DISCESA predizioni consecutive.

CONFERMA_SALITA = 3    # 3 campioni consecutivi per salire (reazione rapida)
CONFERMA_DISCESA = 8   # 8 campioni consecutivi per scendere (inerzia alta)

# Sequenza clinica circolare
SEQUENZA = [0, 1, 2, 3]  # Normale -> Alzando -> Panico -> Abbassando
# Posizione corrente nella sequenza
pos_corrente = 0  # indice in SEQUENZA (0 = Normale)

# Contatori per conferma transizione
contatore_salita = 0
contatore_discesa = 0

# ==========================================
# 1A. CARICAMENTO MODELLO
# ==========================================
try:
    print(f"Caricamento modello da {NOME_MODELLO}...")
    model = load_model(NOME_MODELLO)
    print("Modello caricato.")
except Exception as e:
    print(f"[!] ERRORE FATALE: {e}")
    exit()

# ==========================================
# 1B. SCALER
# ==========================================
print("Calibrazione scaler...")
try:
    df_train = pd.read_csv('training_set.csv', header=None)
    X_train_raw = df_train.iloc[:, colonne_input].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_raw)
    print("Calibrazione completata.")
except FileNotFoundError:
    print("[!] ERRORE: training_set.csv non trovato!")
    exit()

# ==========================================
# 1C. BUFFER
# ==========================================
buffer_dati = deque(maxlen=LOOK_BACK)

# ==========================================
# 2. LETTURA LIVE
# ==========================================
def tail_csv(filepath, colonne):
    with open(filepath, 'r') as f:
        f.seek(0, 2)
        while True:
            linea = f.readline()
            if not linea:
                time.sleep(POLL_INTERVAL)
                continue
            linea = linea.strip()
            if not linea:
                continue
            try:
                valori = [float(x) for x in linea.split(',')]
                riga = [valori[i] for i in colonne]
                yield riga
            except (ValueError, IndexError):
                continue


def il_modello_vuole_salire(prob_media, pos_corrente):
    """
    Determina se il modello suggerisce uno stato PIU' grave
    di quello corrente nella sequenza clinica.
    
    Usa la somma delle probabilità degli stati 'peggiori'
    rispetto alla posizione corrente.
    """
    stato_corrente = SEQUENZA[pos_corrente]
    
    if stato_corrente == 0:  # Normale
        # Vuole salire se prob(alzando) + prob(panico) > prob(normale)
        return (prob_media[1] + prob_media[2]) > prob_media[0]
    elif stato_corrente == 1:  # Alzando
        # Vuole salire se prob(panico) > prob(alzando) + prob(normale)
        return prob_media[2] > (prob_media[1] + prob_media[0]) * 0.5
    elif stato_corrente == 2:  # Panico
        # Siamo al massimo, non si può salire
        return False
    elif stato_corrente == 3:  # Abbassando
        # Da abbassando non si "sale" — si va solo verso normale
        return False
    return False


def il_modello_vuole_scendere(prob_media, pos_corrente):
    """
    Determina se il modello suggerisce uno stato MENO grave.
    """
    stato_corrente = SEQUENZA[pos_corrente]
    
    if stato_corrente == 0:  # Normale
        # Siamo al minimo, non si scende
        return False
    elif stato_corrente == 1:  # Alzando
        # Vuole scendere se prob(normale) > prob(alzando) + prob(panico)
        return prob_media[0] > (prob_media[1] + prob_media[2])
    elif stato_corrente == 2:  # Panico
        # Vuole scendere se prob(panico) NON è più la dominante
        return prob_media[2] < (prob_media[0] + prob_media[1] + prob_media[3])  * 0.4
    elif stato_corrente == 3:  # Abbassando
        # Vuole tornare a normale se prob(normale) è alta
        return prob_media[0] > 0.5
    return False


# ==========================================
# 3. MONITORAGGIO
# ==========================================
print(f"\nIn attesa di dati da {NOME_CSV_LIVE}...")
print(f"Smoothing: {SMOOTHING_WINDOW} | Salita: {CONFERMA_SALITA} conferme | Discesa: {CONFERMA_DISCESA} conferme\n")

ciclo = 0

try:
    for nuova_lettura in tail_csv(NOME_CSV_LIVE, colonne_input):

        buffer_dati.append(nuova_lettura)

        if len(buffer_dati) == LOOK_BACK:
            # --- Predizione ---
            dati_raw = np.array(buffer_dati, dtype=float)
            dati_scalati = scaler.transform(dati_raw)
            X_realtime = np.reshape(dati_scalati, (1, LOOK_BACK, len(colonne_input)))

            predizione_prob = model.predict(X_realtime, verbose=0)[0]

            # --- Smoothing ---
            prob_history.append(predizione_prob)
            prob_media = np.mean(prob_history, axis=0)

            # --- Logica di transizione sequenziale ---
            vuole_salire = il_modello_vuole_salire(prob_media, pos_corrente)
            vuole_scendere = il_modello_vuole_scendere(prob_media, pos_corrente)

            if vuole_salire and pos_corrente < 2:
                # Il modello vuole salire (peggioramento)
                contatore_salita += 1
                contatore_discesa = 0  # reset discesa
                if contatore_salita >= CONFERMA_SALITA:
                    pos_corrente += 1  # sali di UN gradino
                    contatore_salita = 0
                    
            elif vuole_scendere:
                # Il modello vuole scendere (miglioramento)
                contatore_discesa += 1
                contatore_salita = 0  # reset salita
                if contatore_discesa >= CONFERMA_DISCESA:
                    # Da Panico(2) si va ad Abbassando(3)
                    if pos_corrente == 2:
                        pos_corrente = 3  # Panico -> Abbassando
                    # Da Abbassando(3) si torna a Normale(0)
                    elif pos_corrente == 3:
                        pos_corrente = 0  # Abbassando -> Normale
                    # Da Alzando(1) si torna a Normale(0)
                    elif pos_corrente == 1:
                        pos_corrente = 0  # Alzando -> Normale
                    contatore_discesa = 0
            else:
                # Nessuna transizione — reset graduale dei contatori
                if contatore_salita > 0:
                    contatore_salita = max(0, contatore_salita - 1)
                if contatore_discesa > 0:
                    contatore_discesa = max(0, contatore_discesa - 1)

            # --- Display ---
            stato_idx = SEQUENZA[pos_corrente] if pos_corrente < len(SEQUENZA) else 0
            stato_txt = nomi_stati[stato_idx]
            sicurezza = prob_media[stato_idx] * 100

            bpm = nuova_lettura[0]
            hrv = nuova_lettura[1]
            eda = nuova_lettura[2]
            resp = nuova_lettura[3]
            temp = nuova_lettura[4]
            vitals = f"BPM:{bpm:.0f} HRV:{hrv:.0f} EDA:{eda:.1f} Resp:{resp:.0f} T:{temp:.1f}"
            
            # Barra probabilità compatta
            pbar = f"[N:{prob_media[0]*100:.0f} A:{prob_media[1]*100:.0f} P:{prob_media[2]*100:.0f} R:{prob_media[3]*100:.0f}]"

            if stato_idx == 0:
                print(f"[{ciclo:>4}] 🟢 {stato_txt:20s} | {vitals} | {pbar}")
            elif stato_idx == 1:
                print(f"[{ciclo:>4}] 🟡 {stato_txt:20s} | {vitals} | {pbar}")
            elif stato_idx == 2:
                print(f"[{ciclo:>4}] 🔴 {stato_txt:20s} | {vitals} | {pbar} ⚠ GROUNDING")
            else:
                print(f"[{ciclo:>4}] 🔵 {stato_txt:20s} | {vitals} | {pbar}")

        else:
            print(f"Buffer: {len(buffer_dati)}/{LOOK_BACK}", end='\r')

        ciclo += 1

except KeyboardInterrupt:
    print("\n\n--- Monitoraggio interrotto. ---")
