Ecco il contenuto del README.md in italiano, ottimizzato per Obsidian (con l'uso dei Callouts) e completo di tutte le citazioni tecniche basate sui documenti di progetto.🫀 LIMEN — Bracciale Bionico per la Gestione Predittiva dell'Ansia🎯 Il ProblemaGli attacchi di panico colpiscono spesso senza preavviso, ma il corpo manifesta precursori fisiologici diversi minuti prima che l'utente ne diventi consapevole.Durante la finestra di "pre-crisi", il sistema nervoso invia segnali chiari: la variabilità della frequenza cardiaca (HRV) diminuisce, la conduttanza cutanea (EDA) aumenta e la respirazione accelera.I dispositivi wearable attuali rilevano solitamente la frequenza cardiaca elevata solo dopo l'inizio dell'attacco, quando è ormai troppo tardi per intervenire efficacemente.LIMEN è progettato per agire prima del superamento della soglia critica, utilizzando il riconoscimento di pattern biometrici per innescare una de-escalation in tempo reale.💡 La Nostra SoluzioneMonitoraggio Predittivo: Un bracciale intelligente che monitora costantemente 5 segnali biometrici per identificare i pattern pre-crisi con una finestra di circa 60 secondi.Implementazione Edge AI: Esegue una rete neurale LSTM a 2 layer direttamente sul dispositivo utilizzando l'approccio TinyML.Feedback Adattivo: Risponde con un protocollo aptico multi-fase progettato per guidare il corpo verso uno stato di calma, aggirando l'assuefazione dei meccanocettori.Elaborazione Locale: Nessuna dipendenza dal cloud o dallo smartphone; l'inferenza avviene localmente sul microcontrollore per garantire latenza zero e totale privacy dei dati.🏗️ Architettura di SistemaPlaintext┌─────────────────────────────────────────────────────────────────┐
│                        BRACCIALE LIMEN                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │   SENSORI    │    │         CORE EDGE AI (ESP32-S3)      │  │
│  │              │    │                                      │  │
│  │  PPG  ──────────► │  Acquisizione e Filtraggio Segnale   │  │
│  │  EDA  ──────────► │      (Rimozione artefatti IMU)       │  │
│  │  TEMP ──────────► │                  │                   │  │
│  │  IMU  ──────────► │  Finestra scorrevole 60 secondi      │  │
│  │  RESP ──────────► │       (5 feature × 60 tick)          │  │
│  │              │    │                  │                   │  │
│  └──────────────┘    │    LSTM 2-layer (TFLite Micro)       │  │
│                      │    quantizzato a 8-bit · < 150 KB    │  │
│                      │                  │                   │  │
│                      │    ┌─────────────▼──────────────┐    │  │
│                      │    │   Classificazione Stato    │    │  │
│                      │    │  0:Normale  1:Onset        │    │  │
│                      │    │  2:Panico   3:Recovery     │    │  │
│                      │    └─────────────┬──────────────┘    │  │
│                      │                  │                   │  │
│                      │    Loop Aptico Adattivo (val. 40s)   │  │
│                      └──────────────────┼───────────────────┘  │
│                                         │                      │
│  ┌──────────────────────────────────────▼──────────────────┐  │
│  │           3× MOTORI LRA (driver DRV2605L)                │  │
│  │      ●12   ●4    ●8   → pattern spaziali alternati       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
Architettura basata su SoC ESP32-S3 per l'accelerazione delle istruzioni vettoriali.🔬 Componenti HardwareComponenteModelloPosizionamentoFunzionePPGMAX30102 (Ottico)Centro, interno polsoRilevazione HR e HRV (RMSSD) EDA / GSRPiastrine Ag/AgClLati del sensore PPGConduttanza cutanea (sudore) TermometroMAX30205Adiacente a EDAVasocostrizione periferica IMUMPU6050 (6 assi)Scheda madre (PCB)Filtro artefatti e stima respiro Motori Aptici3× LRA + DRV2605LEsterno (ore 12, 4, 8)Feedback tattile adattivo MCU / SoCESP32-S3Scheda madre (PCB)Inferenza Edge + connettività BLE BatteriaLiPo 250 mAhSopra il PCBAutonomia giornaliera Materiali: Scocca in Policarbonato/ABS, rivestimento interno in Silicone Medicale ipoallergenico e cinturino in nylon elastico.Peso: Complessivamente inferiore a 40 grammi per garantire comfort prolungato.🧠 Modello AI — Predittore di Attacchi di PanicoArchitetturaTipo: Rete neurale LSTM (Long Short-Term Memory) a 2 layer.Finestra di Input: 50 timestep × 6 feature (BPM, HRV, EDA, Respirazione, Temperatura, ID Soggetto).Layer:LSTM (16 unità) con Batch Normalization e Dropout (0.4).LSTM (8 unità) con Batch Normalization e Dropout (0.3).Dense (16, ReLU) + Dense (4, Softmax) per la classificazione finale.Addestramento e DeploymentData Augmentation: Iniezione di rumore gaussiano ($\sigma = 0.02$) per raddoppiare il set di dati e migliorare la robustezza.Ottimizzazione: ReduceLROnPlateau dimezza il learning rate se la perdita di validazione non migliora per 5 epoche.Performance: L'accuratezza di validazione converge stabilmente sopra il 97%.Quantizzazione: Modello esportato in ONNX e quantizzato a 8-bit per TensorFlow Lite Micro.Peso: Ridotto a meno di 150 KB per operare direttamente sulla RAM del SoC.💥 Loop Aptico AdattivoIl sistema valuta la risposta fisiologica ogni 40 secondi per evitare l'adattamento neurale (corpuscoli di Pacini).[!NOTE]
Caso A: De-escalation (Il corpo risponde allo stimolo)
* Trigger: Il SoC rileva l'innalzamento dell'HRV e la stabilizzazione dell'EDA.
* Azione: Il pattern aptico rallenta (da 0.3 Hz a 0.15 Hz) e l'intensità sfuma fino a spegnersi.[!WARNING]Caso B: Shift Strategico (L'ansia persiste)Fase 1 (0–40s): Entrainment Respiratorio. Sinusoide lenta per sincronizzare meccanicamente il respiro.Fase 2 (40–80s): Pressione Crescente. Rampa a dente di sega per il grounding somatico.Fase 3 (80–120s): Stimolazione Spaziale Alternata. I 3 motori LRA si attivano in rotazione.Fase 4 (>120s): Pattern Interrupt SOS. Impulsi stile codice Morse per distrarre l'amigdala.📁 Struttura della RepositoryPlaintextlimen/
├── firmware/              # Inferenza TFLite Micro + driver aptici per ESP32-S3
├── ml/
│   ├── attacchiPanico.py  # Pipeline di training LSTM (Batch Norm + LR Reduce)
│   └── electrical_sim.py  # Simulazione hardware del loop di biofeedback
├── data_generation/
│   ├── simulate_realtime.py             # Generatore singolo profilo (P per panico)
│   └── simulate_realtime_multiprofile.py # Generatore multi-profilo (5 archetipi)
├── Dashboard/
│   └── biosignal_plotter.py # Monitor dei biosegnali in tempo reale (stile ECG)
├── data/
│   ├── training_set.csv   # Dati di addestramento generati
│   └── test_set.csv       # Dati di test generati
└── README.md
🚀 Guida Rapida1. Generazione dei Dati SinteticiSimula diversi profili utente (sedentario, atletico, ansioso, ecc.):Bashpython data_generation/simulate_realtime_multiprofile.py --sample_rate 1
2. Addestramento del ModelloAssicurati che i dati generati siano presenti nella cartella data/:Bashpython ml/attacchiPanico.py
3. Dashboard in Tempo RealeVisualizza i biosegnali e gli stati rilevati:Bashpython Dashboard/biosignal_plotter.py --file biosignals_live.csv
👥 Team e ContestoProgetto realizzato per l'Hackathon Elevator Innovation Hub 2026 — "I Nuovi Dispositivi del Futuro".
