import requests #Permite enviar requisições HTTP para a internet
import json #Usada para trabalhar com dados em formato JSON (JavaScript Object Notation)
import time
import datetime
import pytz
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt
import os
import io

# Suprime avisos do TensorFlow para um console mais limpo.
warnings.filterwarnings('ignore')

# --- Variáveis Globais e Configurações ---
CHANNEL_ID = '3068116'
READ_API_KEY = 'OBFR30HBFQH6L9S3'
JANELA_TAMANHO = 1440  # look_back
LIMIAR_ANOMALIA = None
scaler = MinMaxScaler()
model = None
X_train = None  # Variável global para ser usada na conversão

# --- Funções de Análise e Visualização ---
def plot_training_history(history):
    """Plota a perda de treinamento e validação."""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'r.', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('history_training.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(y_true, y_pred, title, subplot_pos):
    """Plota os resíduos de um conjunto de dados."""
    residuals = y_true - y_pred
    
    # Calcular a média e o desvio padrão de cada coluna de resíduos
    residuals_mean = np.mean(residuals, axis=0)
    residuals_std = np.std(residuals, axis=0)
    
    plt.subplot(1, 2, subplot_pos)
    plt.scatter(y_pred[:, 0], residuals[:, 0], c='blue' if subplot_pos == 1 else 'green', 
                marker='o' if subplot_pos == 1 else 's', label='Temp Resíduos')
    plt.scatter(y_pred[:, 1], residuals[:, 1], c='red' if subplot_pos == 1 else 'purple', 
                marker='x' if subplot_pos == 1 else 'd', label='Umid Resíduos')
    
    plt.axhline(y=0, color='k', linestyle='-')
    plt.axhline(y=residuals_mean[0], color='r', linestyle='--', label=f'Mean Temp: {residuals_mean[0]:.3f}')
    plt.axhline(y=residuals_mean[1], color='b', linestyle='--', label=f'Mean Umid: {residuals_mean[1]:.3f}')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True)

def plot_residuals_histogram(residuals, title, subplot_pos):
    """Plota o histograma dos resíduos."""
    residuals_mean = np.mean(residuals, axis=0)
    residuals_std = np.std(residuals, axis=0)

    plt.subplot(1, 2, subplot_pos)
    plt.hist(residuals[:, 0], bins=20, color='blue', alpha=0.6, label='Temp Resíduos')
    plt.hist(residuals[:, 1], bins=20, color='red', alpha=0.6, label='Umid Resíduos')
    
    plt.title(title)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.axvline(x=residuals_mean[0], color='b', linestyle='--', label=f'Mean Temp: {residuals_mean[0]:.3f}')
    plt.axvline(x=residuals_mean[1], color='r', linestyle='--', label=f'Mean Umid: {residuals_mean[1]:.3f}')
    plt.legend(loc='upper right')
    plt.grid(True)

# --- Funções do Modelo e Pré-processamento ---
def preparar_dados_para_treino(data, janela_tamanho):
    """Cria as sequências de entrada (X) e saída (y) para o modelo."""
    X, y = [], []
    for i in range(len(data) - janela_tamanho):
        X.append(data[i:(i + janela_tamanho), :])
        y.append(data[i + janela_tamanho, :])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_shape):
    """
    Constrói um modelo LSTM simples e o compila com otimizador e métricas.
    """
    model = Sequential()
    model.add(Input(shape=input_shape, batch_size=1))
    model.add(LSTM(12, unroll=False, return_sequences=True))
    model.add(LSTM(12, unroll=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def treinar_e_analisar_modelo(caminho_csv):
    global LIMIAR_ANOMALIA, scaler, model, X_train
    
    print("\n Iniciando a Etapa de Treinamento do Modelo")
    try:
        feeds = pd.read_csv(caminho_csv)
        feeds = feeds.drop(['latitude', 'longitude', 'elevation', 'status'], axis=1)
        feeds['created_at'] = pd.to_datetime(feeds['created_at'])
    except FileNotFoundError:
        print("Erro: O arquivo 'feeds.csv' não foi encontrado. Verifique o caminho.")
        return False

    df_limpo = feeds[['created_at', 'field1', 'field2']].rename(columns={'field1': 'Temperatura', 'field2': 'Umidade'})
    df_limpo.set_index('created_at', inplace=True)
    df_limpo = df_limpo.sort_index()

    df_normalizado = scaler.fit_transform(df_limpo) # normalização para valores entre 0 e 1
    
    train_size = int(len(df_normalizado) * 0.8) #len obtém o numero total de linhas
    train_data_scaled = df_normalizado[:train_size] #todas a linhas do inicio até train_size
    test_data_scaled = df_normalizado[train_size:] # de train_size até o fim
    
    X_train, y_train_scaled = preparar_dados_para_treino(train_data_scaled, JANELA_TAMANHO)
    X_test, y_test_scaled = preparar_dados_para_treino(test_data_scaled, JANELA_TAMANHO)

    num_features = df_limpo.shape[1]
    input_shape = (JANELA_TAMANHO, num_features)
    model = build_lstm_model(input_shape, num_features)
    
    print("\n--- Sumário do Modelo ---")
    model.summary() # exibir um resumo da arquitetura do modelo
    
    history = model.fit(
        X_train, y_train_scaled,
        epochs=80,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)], #Se a perda de validação não melhorar por 10 épocas consecutivas, o treinamento é interrompido automaticamente
        validation_split=0.2,
        verbose=1
    )
    
    # Análises de Perda
    plot_training_history(history)

    # Análises de Resíduos
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)
    
    y_train_pred = scaler.inverse_transform(y_train_pred_scaled)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    
    # Obtém os valores reais para comparação com as previsões
    y_train_true = scaler.inverse_transform(y_train_scaled)
    y_test_true = scaler.inverse_transform(y_test_scaled)
    
    plt.figure(figsize=(10, 5))
    plot_residuals(y_train_true, y_train_pred, 'Residuals vs Predicted (Training)', 1)
    plot_residuals(y_test_true, y_test_pred, 'Residuals vs Predicted (Test)', 2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plot_residuals_histogram(y_train_true - y_train_pred, 'Histogram of Residuals (Training)', 1)
    plot_residuals_histogram(y_test_true - y_test_pred, 'Histogram of Residuals (Test)', 2)
    plt.tight_layout()
    plt.show()

    # Define o limiar de anomalia com base nos resíduos do treino
    train_residuals = np.mean(np.power(y_train_scaled - y_train_pred_scaled, 2), axis=1)
    LIMIAR_ANOMALIA = np.percentile(train_residuals, 95)
    print("Treinamento concluído com sucesso!")
    print(f"   Limiar de Anomalia (95º Percentil): {LIMIAR_ANOMALIA:.6f}\n")
    return True

# --- Funções de Conversão para .h  ---
def convert_to_h_file(model, model_name="lstm_model_float"):
    """
    Converte o modelo Keras em um arquivo de cabeçalho C/C++ em formato float32.
    Esta abordagem é mais estável e evita erros de quantização.
    """
    print("Convertendo para TensorFlow Lite (Float32)...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # NÃO adicione otimizações ou tipos de inferência para evitar erros de quantização
    
    tflite_model = converter.convert()
    
    # Geração do arquivo model.h diretamente em Python
    h_file_path = f"{model_name}.h"
    with open(h_file_path, "w") as f:
        f.write("/* Arquivo de modelo TFLite para TinyML */\n\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n")
        f.write(f"const unsigned char g_model[] = {{\n")
        
        # Escreve os bytes em formato de array C++
        hex_line = []
        for byte in tflite_model:
            hex_line.append(f"0x{byte:02x}")
            if len(hex_line) == 12:
                f.write("  " + ", ".join(hex_line) + ",\n")
                hex_line = []
        if hex_line:
            f.write("  " + ", ".join(hex_line) + "\n")
            
        f.write("};\n")

    print(f"Arquivo de cabeçalho C/C++ salvo em: {h_file_path}")
    
    # Exibe informações do modelo float32
    print("\n--- Informações para o Microcontrolador (Float32) ---")
    print(f"Modelo salvo em formato float32. Não há informações de quantização.")
    print(f"Scaler Min: {scaler.data_min_}")
    print(f"Scaler Max: {scaler.data_max_}")

def obter_e_analisar_dados_em_tempo_real():
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={JANELA_TAMANHO + 1}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        feeds = data.get('feeds', [])
        
        if len(feeds) < JANELA_TAMANHO + 1:
            print(f"Aviso: Dados insuficientes para análise. Necessário {JANELA_TAMANHO + 1} pontos.")
            return

        df_real_time = pd.DataFrame(feeds)
        df_real_time['created_at'] = pd.to_datetime(df_real_time['created_at'])
        df_real_time['field1'] = pd.to_numeric(df_real_time['field1'], errors='coerce')
        df_real_time['field2'] = pd.to_numeric(df_real_time['field2'], errors='coerce')
        
        df_real_time = df_real_time[['created_at', 'field1', 'field2']].rename(columns={'field1': 'Temperatura', 'field2': 'Umidade'})
        df_real_time.set_index('created_at', inplace=True)
        df_real_time = df_real_time.sort_index()

        if df_real_time.isnull().values.any():
            print("Aviso: Dados recebidos contêm valores não numéricos. Pulando esta leitura.")
            return

        janela_normalizada = scaler.transform(df_real_time.iloc[:-1])
        ponto_real_normalizado = scaler.transform(df_real_time.iloc[-1:])

        janela_input = np.reshape(janela_normalizada, (1, JANELA_TAMANHO, 2))
        
        previsao = model.predict(janela_input)
        
        erro_previsao = np.mean(np.power(ponto_real_normalizado - previsao[0], 2))
        
        ponto_recente = df_real_time.iloc[-1]
        fuso_horario = pytz.timezone('America/Sao_Paulo')
        data_e_hora_local = datetime.datetime.now(fuso_horario)

        print("\n--- Análise em Tempo Real ---")
        print(f" Data e Hora (ThingSpeak): {ponto_recente.name.strftime('%Y-%m-%d %H:%M:%S')}")
        print(data_e_hora_local.strftime('%d/%m/%Y %H:%M:%S'))
        print(f" Temperatura: {ponto_recente['Temperatura']:.2f}°C")
        print(f" Umidade: {ponto_recente['Umidade']:.2f}%")
        print(f" Erro de Previsão: {erro_previsao:.6f}") 
        print(f" Limiar de Anomalia: {LIMIAR_ANOMALIA:.6f}")

        if erro_previsao > LIMIAR_ANOMALIA:
            print("\n ALERTA: ANOMALIA DETECTADA!")
            print("A leitura está fora do padrão normal previsto")
            
            sugestao_real = scaler.inverse_transform(previsao)[0]
            print(f"Sugestão do Modelo: Temp: {sugestao_real[0]:.2f}°C, Umid: {sugestao_real[1]:.2f}%\n")
        else:
            print("\n Sem Anomalias (Leitura dentro do padrão esperado)")

    except requests.exceptions.RequestException as e:
        print(f"Erro de Conexão com ThingSpeak: {e}")
    except json.JSONDecodeError:
        print("Erro: Não foi possível decodificar a resposta JSON.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    if treinar_e_analisar_modelo('C:/Users/UFMA/Downloads/feeds.csv'):
        convert_to_h_file(model)
        
        print("\nIniciando o Monitoramento Contínuo\n" + "-"*40)
        while True:
            obter_e_analisar_dados_em_tempo_real()
            print("\n" + "-"*40)
            print("Aguardando 15 segundos para a próxima leitura...")
            time.sleep(60)