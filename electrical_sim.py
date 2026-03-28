# Simulazione Hardware di Limen

import matplotlib.pyplot as plt
import random

# Parametri di simulazione
cycles = 25
evaluation_interval = 40

# Liste per tracciare lo storico
history_time = []
history_hrv = []
history_anxiety = []
history_phase = []

# Condizioni iniziali (Utente sotto stress)
current_hrv = 25.0 
anxiety_timer = 0
eda_stable = False

# Loop di simulazione dinamico
for i in range(cycles):
    time_elapsed = i * evaluation_interval
    history_time.append(time_elapsed)
    
    # Logica di valutazione (come descritta nei documenti)
    if current_hrv > 50 and eda_stable:
        state = "Caso A"
    else:
        state = "Caso B"
        
    # Registra i dati attuali
    history_hrv.append(current_hrv)
    history_anxiety.append(anxiety_timer)
    
    # Attuazione e reazione simulata del corpo umano
    if state == "Caso A":
        history_phase.append("De-escalation")
        anxiety_timer = 0
        # Il corpo è rilassato, l'HRV fluttua positivamente
        current_hrv += random.uniform(-2, 5)
        # L'EDA rimane prevalentemente stabile
        eda_stable = random.choice([True, True, True, False]) 
    else:
        # L'ansia persiste, scaliamo le fasi
        if anxiety_timer < 40:
            history_phase.append("Fase 1: Entrainment")
            current_hrv += random.uniform(-3, 6) # Lieve miglioramento
        elif anxiety_timer < 80:
            history_phase.append("Fase 2: Pressione")
            current_hrv += random.uniform(-2, 8)
        elif anxiety_timer < 120:
            history_phase.append("Fase 3: Spaziale")
            current_hrv += random.uniform(4, 12) # Inizia a sbloccarsi
        else:
            history_phase.append("Fase 4: SOS")
            current_hrv += random.uniform(10, 20) # Intervento forte, rottura del loop
        
        anxiety_timer += evaluation_interval
        
        # Simula la stabilizzazione dell'EDA al crescere dell'HRV
        if current_hrv > 45:
            eda_stable = random.choice([True, False])
        if current_hrv > 55:
            eda_stable = True
            
    # Manteniamo l'HRV in un range realistico (0-100)
    current_hrv = min(100, max(0, current_hrv))

# Creazione del Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Subplot 1: Andamento HRV
ax1.plot(history_time, history_hrv, marker='o', color='dodgerblue', linewidth=2, label='Livello HRV (Simulato)')
ax1.axhline(y=50, color='crimson', linestyle='--', linewidth=1.5, label='Soglia De-escalation (>50)')
ax1.set_ylabel('Valore HRV', fontsize=11)
ax1.set_title('Simulazione Biofeedback LIMEN: HRV e Risposta Fisiologica', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.7)

# Subplot 2: Fasi di Intervento Aptico e Timer Ansia
bars = ax2.bar(history_time, history_anxiety, width=25, color='darkorange', alpha=0.8, label='Timer Ansia Persistente (s)')
ax2.set_ylabel('Timer Ansia (s)', fontsize=11)
ax2.set_xlabel('Tempo Trascorso (Secondi)', fontsize=11)
ax2.set_title('Progressione Interventi Aptici (Loop Adattivo)', fontsize=12)
ax2.grid(axis='y', linestyle=':', alpha=0.7)

# Annotazione delle Fasi sui grafici
last_phase = ""
for i, txt in enumerate(history_phase):
    # Annotiamo solo quando la fase cambia per non sovrapporre il testo
    if txt != last_phase or txt == "De-escalation":
        # Posizioniamo l'etichetta sopra le barre
        y_pos = history_anxiety[i] + 5 if txt != "De-escalation" else 10
        color = 'green' if txt == "De-escalation" else 'black'
        ax2.annotate(txt, (history_time[i], y_pos), 
                     xytext=(0, 5), textcoords="offset points", 
                     ha='center', va='bottom', fontsize=9, rotation=45, color=color, fontweight='bold')
        last_phase = txt

plt.tight_layout()
plt.show() # Mostra il grafico a schermo