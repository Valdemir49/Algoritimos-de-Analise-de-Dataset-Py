import requests
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import warnings

# --- 1. CONFIGURAÇÕES GLOBAIS ---
# Suas chaves e parâmetros, definidos aqui para fácil acesso.
CHANNEL_ID = '3068116'
READ_API_KEY = 'OBFR30HBFQH6L9S3'
JANELA_TAMANHO = 10
LIMIAR_ANOMALIA = None
scaler = MinMaxScaler()
model = None

# Suprime avisos do TensorFlow para um console mais limpo.
warnings.filterwarnings('ignore')

# --- 2. FUNÇÕES DE DEEP LEARNING ---

def preparar_dados(df, janela_tamanho):
    """
    Cria janelas de tempo (sliding windows) para o modelo LSTM.
    """
    janelas = []
    for i in range(len(df) - janela_tamanho):
        janelas.append(df.iloc[i:i + janela_tamanho].values)
    return np.array(janelas)

def build_model(janela_tamanho, num_features):
    """
    Constrói a arquitetura do Autoencoder Recorrente.
    """
    inputs = Input(shape=(janela_tamanho, num_features))
    
    # Camada Encoder para aprender a representação da sequência
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)
    
    # Camada Decoder para reconstruir a sequência original
    decoded = tf.keras.layers.RepeatVector(janela_tamanho)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    
    # Camada de saída para gerar a previsão
    outputs = TimeDistributed(Dense(num_features))(decoded)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def treinar_modelo(caminho_csv):
    """
    Carrega o dataset, treina o modelo e define o limiar de anomalia.
    Retorna True se o treinamento for bem-sucedido, False caso contrário.
    """
    global LIMIAR_ANOMALIA, scaler, model
    
    print("\n📦 **Iniciando a Etapa de Treinamento do Modelo**")
    try:
        feeds = pd.read_csv(caminho_csv)
        feeds = feeds.drop(['latitude', 'longitude', 'elevation', 'status'], axis=1)
        feeds['created_at'] = pd.to_datetime(feeds['created_at'])
    except FileNotFoundError:
        print("❌ Erro: O arquivo 'feeds.csv' não foi encontrado. Verifique o caminho.")
        return False

    df_limpo = feeds[['created_at', 'field1', 'field2']].rename(columns={'field1': 'Temperatura', 'field2': 'Umidade'})
    df_limpo.set_index('created_at', inplace=True)
    df_limpo = df_limpo.sort_index()

    # Normaliza os dados para o treinamento
    df_normalizado = scaler.fit_transform(df_limpo)
    
    # Divide os dados para treino (80%) e teste (20%)
    train_size = int(len(df_normalizado) * 0.8)
    train_data = df_normalizado[:train_size]
    
    X_train = preparar_dados(pd.DataFrame(train_data), JANELA_TAMANHO)
    y_train = X_train

    # Constrói e treina o modelo
    num_features = df_limpo.shape[1]
    model = build_model(JANELA_TAMANHO, num_features)
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)],
        verbose=0
    )

    # Define o limiar de anomalia com base nos erros de reconstrução do treino
    train_previsoes = model.predict(X_train)
    train_mse = np.mean(np.power(X_train - train_previsoes, 2), axis=(1, 2))
    
    LIMIAR_ANOMALIA = np.percentile(train_mse, 95)
    print("✅ Treinamento concluído com sucesso!")
    print(f"   Limiar de Anomalia (95º Percentil): {LIMIAR_ANOMALIA:.6f}\n")
    return True

# --- 3. FUNÇÃO DE DETECÇÃO EM TEMPO REAL ---

def obter_e_analisar_dados_em_tempo_real():
    """
    Faz a requisição dos dados mais recentes e realiza a detecção de anomalia.
    """
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={JANELA_TAMANHO}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        feeds = data.get('feeds', [])
        
        # Garante que há dados suficientes para a janela de tempo
        if len(feeds) < JANELA_TAMANHO:
            print(f"⚠️ Aviso: Dados insuficientes para análise. Necessário {JANELA_TAMANHO} pontos.")
            return

        df_real_time = pd.DataFrame(feeds)
        df_real_time['created_at'] = pd.to_datetime(df_real_time['created_at'])
        
        # Converte as colunas para numérico, tratando erros
        df_real_time['field1'] = pd.to_numeric(df_real_time['field1'], errors='coerce')
        df_real_time['field2'] = pd.to_numeric(df_real_time['field2'], errors='coerce')
        
        df_real_time = df_real_time[['created_at', 'field1', 'field2']].rename(columns={'field1': 'Temperatura', 'field2': 'Umidade'})
        df_real_time.set_index('created_at', inplace=True)
        df_real_time = df_real_time.sort_index()

        if df_real_time.isnull().values.any():
            print("⚠️ Aviso: Dados recebidos contêm valores não numéricos. Pulando esta leitura.")
            return
        
        ponto_recente = df_real_time.iloc[-1]
        janela_normalizada = scaler.transform(df_real_time)
        janela_input = np.reshape(janela_normalizada, (1, JANELA_TAMANHO, 2))

        # Reconstrução e cálculo do erro
        reconstrucao = model.predict(janela_input)
        erro_reconstrucao = np.mean(np.power(janela_normalizada - reconstrucao[0], 2))

        print("\n--- 🔎 Análise em Tempo Real ---")
        print(f"   **Data e Hora:** {ponto_recente.name.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   **Temperatura:** {ponto_recente['Temperatura']:.2f}°C")
        print(f"   **Umidade:** {ponto_recente['Umidade']:.2f}%")
        print(f"   **Erro de Reconstrução:** {erro_reconstrucao:.6f}")
        print(f"   **Limiar de Anomalia:** {LIMIAR_ANOMALIA:.6f}")

        if erro_reconstrucao > LIMIAR_ANOMALIA:
            print("\n🚨 **ALERTA: ANOMALIA DETECTADA!**")
            print("   A leitura está fora do padrão normal.")
            
            sugestao_normalizada = reconstrucao[0][-1]
            dummy_array = np.zeros((1, 2))
            dummy_array[0, :] = sugestao_normalizada
            sugestao_real = scaler.inverse_transform(dummy_array)[0]
            print(f"   **Sugestão do Modelo:** Temp: {sugestao_real[0]:.2f}°C, Umid: {sugestao_real[1]:.2f}%")
        else:
            print("\n✅ **Sem Anomalias.** Leitura dentro do padrão esperado.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de Conexão com ThingSpeak: {e}")
    except json.JSONDecodeError:
        print("❌ Erro: Não foi possível decodificar a resposta JSON.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")

# --- 4. LOOP PRINCIPAL ---
if __name__ == "__main__":
    if treinar_modelo('C:/Users/Admin/Downloads/feeds.csv'):
        print("\n✨ **Iniciando o Monitoramento Contínuo**\n" + "-"*40)
        while True:
            obter_e_analisar_dados_em_tempo_real()
            print("\n" + "-"*40)
            print("Aguardando 15 segundos para a próxima leitura...")
            time.sleep(15)