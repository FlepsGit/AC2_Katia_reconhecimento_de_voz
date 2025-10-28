import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from gtts import gTTS
from IPython.display import Audio, display

# ===============================
#  CONFIGURAÇÃO DO TEMA POKÉMON
# ===============================
commands = ['Ataque', 'Capturar', 'Corra', 'Mega_evolua', 'Usar_item']
data_path = "audios_pokemon"  # pasta com as subpastas de cada comando


# ===============================
#  FUNÇÃO PARA EXTRAIR MFCCs
# ===============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=2.5, offset=0.5, sr=22050)
    y, _ = librosa.effects.trim(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
    return np.mean(mfccs.T, axis=0)



# ===============================
#  CARREGAR E PROCESSAR OS DADOS
# ===============================
features, labels = [], []
for i, label in enumerate(commands):
    folder = os.path.join(data_path, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            data = extract_features(path)
            features.append(data)
            labels.append(i)

features = np.array(features)
labels = to_categorical(np.array(labels))

# ===============================
#  MOSTRAR GRÁFICO DE MFCCs
# ===============================
grafico_comando = int(input("Qual comando deseja ver o gráfico de MFCCS? (número da ordem em que aparece): "))
grafico_comando = grafico_comando - 1 
example_file = os.path.join(data_path, commands[grafico_comando], os.listdir(os.path.join(data_path, commands[grafico_comando]))[grafico_comando])
y, sr = librosa.load(example_file, duration=2.5, offset=0.5)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar()
plt.title(f"Exemplo de MFCCs - {commands[grafico_comando].capitalize()}")
plt.tight_layout()
plt.show()

# ===============================
#  TREINO E TESTE
# ===============================
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,train_size=0.8, random_state=42)

model = Sequential([
    Dense(512, activation='relu', input_shape=(60,)),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(commands), activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# ===============================
#  AVALIAÇÃO DO MODELO
# ===============================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Acurácia do modelo: {acc*100:.2f}%")

plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Teste')
plt.title('Evolução da Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
# ===============================
#  FUNÇÃO PARA PREDIZER COMANDO
# ===============================
def predict_audio(file_path):
    feat = extract_features(file_path)
    feat = np.expand_dims(feat, axis=0)
    pred = model.predict(feat)
    command = commands[np.argmax(pred)]
    print(f"\n🔊 Comando detectado: {command.upper()} (confiança: {np.max(pred)*100:.1f}%)")

# ===============================
#  TESTAR COM VOZ 
# ===============================
print("Testar com a sua voz")
choice = input("Digite 1 para testar: ")
if choice == "1":
    duration = 3  # segundos
    print("🎙️ Gravando... fale o comando agora!")
    fs = 22050
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, fs, (audio * 32767).astype(np.int16))
    print("✅ Gravação concluída.")
    predict_audio(temp_file.name)