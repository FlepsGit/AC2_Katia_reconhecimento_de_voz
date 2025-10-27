import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from IPython.display import Audio, display
import os
import time

# ------------------------
# CONFIGURAÇÕES INICIAIS
# ------------------------
DATASET_PATH = "audios_pokemon"
N_MFCC = 13
DURATION = 2.0   # ajuste conforme a média dos seus áudios
SR = 22050

# ------------------------
# CARREGAR OS ÁUDIOS
# ------------------------
def carregar_dataset(dataset_path=DATASET_PATH):
    comandos = sorted(os.listdir(dataset_path))
    X, y, caminhos = [], [], []
    print(f"Comandos encontrados: {comandos}\n")

    for idx, comando in enumerate(comandos):
        pasta = os.path.join(dataset_path, comando)
        if not os.path.isdir(pasta):
            continue

        for arquivo in os.listdir(pasta):
            if not arquivo.endswith(".wav"):
                continue
            caminho = os.path.join(pasta, arquivo)
            audio, sr = librosa.load(caminho, sr=SR, duration=DURATION)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            mfcc_mean = mfcc.mean(axis=1)
            X.append(mfcc_mean)
            y.append(idx)
            caminhos.append(caminho)

    X = np.array(X)
    y = np.array(y)
    print(f"Total de amostras carregadas: {len(X)}")
    return X, y, comandos, caminhos

# ------------------------
# TREINO E TESTE
# ------------------------
def treinar_modelo():
    X, y, comandos, caminhos = carregar_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test, caminhos_train, caminhos_test = train_test_split(
        X, y, caminhos, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Amostras de treino: {len(X_train)}")
    print(f"Amostras de teste:  {len(X_test)}")

    model = keras.Sequential([
        layers.Dense(64, input_dim=N_MFCC, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(len(set(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\nTreinando o modelo...")
    history = model.fit(X_train, y_train, epochs=200, verbose=0)
    print("✅ Treinamento concluído!")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia no conjunto de TESTE: {acc:.2f}\n")

    pred = model.predict(X_test)

    print("🔊 Testando o modelo com reprodução dos áudios:\n")
    for i, p in enumerate(pred):
        caminho = caminhos_test[i]
        classe_predita = comandos[np.argmax(p)]
        confianca = np.max(p)

        print(f"▶️ Áudio {i+1}/{len(X_test)}: {os.path.basename(caminho)}")
        display(Audio(caminho, rate=SR))  # toca o áudio
        print(f"🔹 Predição: {classe_predita} (confiança: {confianca:.2f})\n")

        time.sleep(2.5)  # pausa para não avançar antes do áudio terminar

    plt.plot(history.history["accuracy"])
    plt.title("Evolução da Acurácia no Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.show()

    model.save("modelo_comandos.h5")
    np.save("comandos.npy", np.array(comandos))
    print("\nModelo e classes salvos com sucesso!")

# ------------------------
# EXECUTAR
# ------------------------

treinar_modelo()