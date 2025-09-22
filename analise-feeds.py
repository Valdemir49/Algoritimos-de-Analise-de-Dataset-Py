import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Funções de Análise e Visualização ---

def analise_estatistica_e_grafica(df, coluna, nome_coluna, unidade, nome_periodo):
    """
    Realiza análise estatística e gera um gráfico de distribuição para uma coluna.
    """
    if df.empty:
        print(f"O grupo de dados da {nome_periodo} está vazio. Não há dados para analisar.")
        return

    media = df[coluna].mean()
    desvio_padrao = df[coluna].std()
    
    print(f"--- Análise de {nome_coluna} na {nome_periodo} ---")
    print(f"Média: {media:.2f}{unidade}")
    print(f"Desvio Padrão: {desvio_padrao:.2f}{unidade}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[coluna], kde=True)
    plt.title(f'Distribuição de {nome_coluna} na {nome_periodo}', fontsize=16)
    plt.xlabel(f'{nome_coluna} ({unidade})', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.show()

def analise_comportamento(df, colunas_a_plotar, titulo_extra=""):
    """
    Gera um gráfico de linha para visualizar a variação de múltiplas colunas ao longo do tempo.
    """
    if df.empty:
        print("DataFrame está vazio. Não há dados para plotar.")
        return

    plt.figure(figsize=(15, 8))
    
    for coluna, rotulo in colunas_a_plotar.items():
        sns.lineplot(x='created_at', y=coluna, data=df, label=rotulo)
    
    plt.title(f'Variação das Condições Ambientais ao Longo do Dia {titulo_extra}', fontsize=16)
    plt.xlabel('Hora do Dia', fontsize=12)
    plt.ylabel('Valores', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Lógica do Menu Interativo (Versão Final) ---
def main():
    
    try:
        feeds = pd.read_csv('C:/Users/Admin/Downloads/feeds.csv')
        feeds = feeds.drop(['latitude', 'longitude', 'elevation', 'status'], axis=1)
        feeds['created_at'] = pd.to_datetime(feeds['created_at'])
    except FileNotFoundError:
        print("Erro: O arquivo 'feeds.csv' não foi encontrado. Verifique o caminho.")
        return

    feeds['data'] = feeds['created_at'].dt.date
    feeds['hora'] = feeds['created_at'].dt.hour
    
    while True:
        # Bloco de informação dos dias movido para o início do loop
        dias_unicos = feeds['data'].unique()
        dias_completos = []
        dias_incompletos = []
        
        for dia in dias_unicos:
            df_do_dia = feeds[feeds['data'] == dia].copy()
            horas_presentes = df_do_dia['hora'].nunique()
            if horas_presentes == 24:
                dias_completos.append(dia.strftime('%d/%m/%Y'))
            else:
                dias_incompletos.append(dia.strftime('%d/%m/%Y'))
        
        print("\n--- Análise de Dados do Sensor - Menu ---")
        print(f"Total de dias no dataset: {len(dias_unicos)}")
        print(f"  - Completos: {', '.join(dias_completos)}")
        print(f"  - Incompletos: {', '.join(dias_incompletos)}")
        
        print("\nEscolha uma opção:")
        print("1. Analisar um dia específico")
        print("2. Exportar um dia para CSV")
        print("3. Sair")
        
        escolha = input("Digite o número da sua escolha: ")
        
        if escolha == '1':
            data_str = input("Digite a data para análise (formato aaaa-mm-dd): ")
            try:
                data_escolhida = pd.to_datetime(data_str).date()
            except ValueError:
                print("Formato de data inválido. Use aaaa-mm-dd.")
                continue

            if data_escolhida not in dias_unicos:
                print("Data não encontrada no dataset. Tente novamente.")
                continue

            df_do_dia = feeds[feeds['data'] == data_escolhida].copy()
            
            grupos_periodo = {
                'Manhã': df_do_dia[(df_do_dia['hora'] >= 6) & (df_do_dia['hora'] <= 12)].copy(),
                'Tarde': df_do_dia[(df_do_dia['hora'] >= 13) & (df_do_dia['hora'] <= 17)].copy(),
                'Noite': df_do_dia[(df_do_dia['hora'] >= 18) & (df_do_dia['hora'] <= 23)].copy(),
                'Madrugada': df_do_dia[(df_do_dia['hora'] >= 0) & (df_do_dia['hora'] <= 5)].copy()
            }
            
            print(f"\n--- Análise detalhada para o dia {data_escolhida.strftime('%d/%m/%Y')} ---")
            for nome_grupo, df_grupo in grupos_periodo.items():
                if not df_grupo.empty:
                    analise_estatistica_e_grafica(df_grupo, 'field1', 'Temperatura', '°C', nome_grupo)
                    analise_estatistica_e_grafica(df_grupo, 'field2', 'Umidade', '%', nome_grupo)
            
            analise_comportamento(df_do_dia, {'field1': 'Temperatura (°C)', 'field2': 'Umidade (%)'}, f"de {data_escolhida.strftime('%d/%m/%Y')}")

        elif escolha == '2':
            data_str = input("Digite a data para exportar (formato aaaa-mm-dd): ")
            try:
                data_escolhida = pd.to_datetime(data_str).date()
            except ValueError:
                print("Formato de data inválido. Use aaaa-mm-dd.")
                continue

            if data_escolhida not in dias_unicos:
                print("Data não encontrada no dataset. Tente novamente.")
                continue
                
            df_a_exportar = feeds[feeds['data'] == data_escolhida].copy()
            nome_arquivo = f'dados_{data_escolhida}.csv'
            df_a_exportar.to_csv(nome_arquivo, index=False)
            print(f"Dados do dia {data_escolhida.strftime('%d/%m/%Y')} exportados para '{nome_arquivo}'.")

        elif escolha == '3':
            print("Saindo do programa. Até a próxima!")
            break

        else:
            print("Opção inválida. Por favor, digite 1, 2 ou 3.")

# Executar o programa
if __name__ == "__main__":
    main()