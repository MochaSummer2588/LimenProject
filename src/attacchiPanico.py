import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
# MODIFICA: Aggiunto BatchNormalization
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization 
# MODIFICA: Aggiunto ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers 
import matplotlib.pyplot as plt

# ==========================================
# 0. RESET TOTALE DELLA MEMORIA
# ==========================================
K.clear_session()

# ==========================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
print("Aggiornamento Protocollo SINTESI: Avvio sessione STABILIZZATA (Batch Norm + LR Reduce)...")

df_train = pd.read_csv('training_set.csv', header=None)
df_test = pd.read_csv('test_set.csv', header=None)

# ORDINE COLONNE: 0:bpm, 1:hrv, 2:eda, 3:resp, 4:temp, 5:state, 6:event, 7:subject
colonne_input = [0, 1, 2, 3, 4, 7] 
colonna_target = 5                 
colonna_subject = 7 

X_train_raw = df_train.iloc[:, colonne_input].values
y_train_raw = df_train.iloc[:, colonna_target].values
subjects_train = df_train.iloc[:, colonna_subject].values 

X_test_raw = df_test.iloc[:, colonne_input].values
y_test_raw = df_test.iloc[:, colonna_target].values
subjects_test = df_test.iloc[:, colonna_subject].values

# MODIFICA: Controllo bilanciamento classi
print("\n--- ANALISI DATI ---")
print("Distribuzione classi nel training originale:")
for stato, conteggio in zip(*np.unique(y_train_raw, return_counts=True)):
    print(f"Stato {int(stato)}: {conteggio} campioni")
print("--------------------\n")

print(f"Feature in ingresso rilevate: {X_train_raw.shape[1]} (Sensori + ID Cliente)")

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

def create_dataset_by_subject(X, y, subjects, look_back):
    X_seq, y_seq = [], []
    unique_subjects = np.unique(subjects) 
    
    for subj in unique_subjects:
        idx = np.where(subjects == subj)[0]
        X_subj = X[idx]
        y_subj = y[idx]
        
        for i in range(len(X_subj) - look_back):
            X_seq.append(X_subj[i:(i + look_back), :])
            y_seq.append(y_subj[i + look_back])
            
    return np.array(X_seq), np.array(y_seq)

LOOK_BACK = 50
print(f"Creazione finestre temporali isolate per soggetto...")
X_train, y_train = create_dataset_by_subject(X_train_scaled, y_train_raw, subjects_train, LOOK_BACK)
X_test, y_test = create_dataset_by_subject(X_test_scaled, y_test_raw, subjects_test, LOOK_BACK)

# ---------------------------------------------------------
# DATA AUGMENTATION 
# ---------------------------------------------------------
print("Esecuzione Data Augmentation...")
noise_factor = 0.02 
X_train_noisy = X_train + np.random.normal(0, noise_factor, X_train.shape)

X_train_final = np.concatenate([X_train, X_train_noisy])
y_train_final = np.concatenate([y_train, y_train])

print(f"Dataset espanso: da {len(X_train)} a {len(X_train_final)} campioni validi.")

# ==========================================
# 2. MODELLO NEURALE (STABILIZZATO PRO)
# ==========================================
print("Costruzione modello con Batch Normalization...")
model = Sequential([
    # Primo strato LSTM con return_sequences=True per passare i dati al prossimo livello LSTM
    LSTM(16, 
         input_shape=(LOOK_BACK, len(colonne_input)),
         return_sequences=True,
         kernel_initializer='glorot_uniform', 
         recurrent_initializer='orthogonal',
         kernel_regularizer=regularizers.l2(0.001)), 
    
    BatchNormalization(), # <-- STABILIZZATORE 1
    Dropout(0.4), 
    
    # Secondo strato LSTM per catturare pattern più complessi
    LSTM(8, kernel_regularizer=regularizers.l2(0.001)),
    
    BatchNormalization(), # <-- STABILIZZATORE 2
    Dropout(0.3),
    
    Dense(16, 
          activation='relu',
          kernel_initializer='he_normal',
          kernel_regularizer=regularizers.l2(0.001)),
    
    Dense(4, 
          activation='softmax',
          kernel_initializer='glorot_uniform')
])

optimizer_custom = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer_custom, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. ADDESTRAMENTO
# ==========================================
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# MODIFICA: Se la Loss del test smette di migliorare per 5 epoche, dimezza la velocità di apprendimento
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

print("Inizio addestramento ottimizzato...")
history = model.fit(
    X_train_final, y_train_final, 
    epochs=100, 
    batch_size=64, 
    validation_data=(X_test, y_test), 
    # MODIFICA: Aggiunto reduce_lr alla lista dei callbacks
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# --- VISUALIZZAZIONE ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Test (Validazione)')
plt.title('Accuratezza (SINTESI)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Test (Validazione)')
plt.title('Perdita (Cross-Entropy)')
plt.legend()
plt.show()

print("Sessione volatile completata. Parametri non salvati su disco.")

# ==========================================
# 4. TEST SUI DATI
# ==========================================
print("\n--- VALUTAZIONE SUI DATI DI TEST ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuratezza totale sul test set: {accuracy * 100:.2f}%\n")

print("--- DETTAGLIO PRIME 5 PREDIZIONI ---")
campioni_da_mostrare = 5
predizioni_prob = model.predict(X_test[:campioni_da_mostrare])
predizioni_classi = np.argmax(predizioni_prob, axis=1)

nomi_stati = ["Normale", "Si sta alzando", "Panico", "Si sta abbassando"]

for i in range(campioni_da_mostrare):
    reale_idx = int(y_test[i])
    predetto_idx = predizioni_classi[i]
    sicurezza = predizioni_prob[i][predetto_idx] * 100
    
    stato_reale = nomi_stati[reale_idx] if 0 <= reale_idx < len(nomi_stati) else f"Sconosciuto ({reale_idx})"
    stato_predetto = nomi_stati[predetto_idx]
        
    print(f"Campione {i+1}:")
    print(f"  - Stato Predetto: {stato_predetto} (Sicurezza: {sicurezza:.1f}%)")
    print(f"  - Stato Reale nel file: {stato_reale}")
    if predetto_idx == reale_idx:
        print("  - [X] PREVISIONE CORRETTA")
    else:
        print("  - [ ] PREVISIONE ERRATA")
    print("-" * 40)