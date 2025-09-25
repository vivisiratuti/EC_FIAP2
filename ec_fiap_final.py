# ------------------------------------------------------------
# PREPARAÇÃO DE BIBLIOTECAS, DADOS E DASHBOARD
# ------------------------------------------------------------
# Importa as bibliotecas necessárias
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import datetime
import numpy as np
import altair as alt
import gdown

# Configura a página do Streamlit
st.set_page_config(layout="wide", page_title="Dashboard RouteMind",
                   page_icon="RouteMind/Assets/Images/Logo_RouteMind.png")

# Define a cor de fundo com base na identidade visual
backgroundColor = "#F0F2F6"

st.markdown(
    f"""
    <style>
    .stApp {{
    background-color: {backgroundColor};
    }}
    </style>
    """,
    unsafe_allow_html=True)

# Layout do cabeçalho
col1, col2 = st.columns([1, 5], gap="small", vertical_alignment="center")

with col1:
    st.image("Assets/Images/Logo_RouteMind.png", width=75)

with col2:
    st.header("Mapeamento Inteligente de Rotas e Comportamentos")

# Carrega e prepara os dados
try:
    # Define a URL e o nome do arquivo de saída
    url = "https://drive.google.com/uc?id=1HUS0Yk9DiY0FfZrnAq8FoxCvhtp3nMK9"
    output_filename = 'df_t.csv'

    # Usa o gdown para baixar o arquivo
    gdown.download(url, output_filename, quiet=False)

    # Lê o arquivo que foi baixado localmente
    df = pd.read_csv(output_filename)

    # Limpa os espaços
    df.columns = df.columns.str.strip()
 
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()
 
# Renomeia colunas para um nome mais amigável
df.rename(columns={
    "nk_ota_localizer_id": "ID_VENDA",
    "fk_contact": "ID_CLIENTE",
    "date_purchase": "DATA_COMPRA",
    "time_purchase": "HORA_COMPRA",
    "place_origin_departure": "LOCAL_PARTIDA_IDA",
    "place_destination_departure": "LOCAL_DESTINO_IDA",
    "place_origin_return": "LOCAL_PARTIDA_VOLTA",
    "place_destination_return": "LOCAL_DESTINO_VOLTA",
    "fk_departure_ota_bus_company": "EMPRESA_IDA",
    "fk_return_ota_bus_company": "EMPRESA_VOLTA",
    "gmv_success": "VALOR_TICKET",
    "total_tickets_quantity_success": "QUANTIDADE_TICKETS"
}, inplace=True)

# Converte tipos necessários
df['DATA_COMPRA'] = pd.to_datetime(df['DATA_COMPRA'], errors='coerce')
df['VALOR_TICKET'] = pd.to_numeric(df['VALOR_TICKET'], errors='coerce')
df['HORA_COMPRA'] = pd.to_datetime(df['HORA_COMPRA'], format='%H:%M:%S', errors='coerce').dt.time

# Filtra dados recentes para o dashboard não ficar pesado
df_filtrado_por_ano = df[(df['DATA_COMPRA'].dt.year >= 2020) & (df['DATA_COMPRA'].dt.year <= 2024)]

# Cria IDs simplificados para clientes e destinos
df_filtrado_por_ano['ID_CLIENTE_SIMPLIFICADO'] = pd.factorize(df_filtrado_por_ano['ID_CLIENTE'])[0] + 1
df_filtrado_por_ano['LOCAL_DESTINO_IDA_SIMPLIFICADO'] = pd.factorize(df_filtrado_por_ano['LOCAL_DESTINO_IDA'])[0] + 1

# ------------------------------------------------------------
# CÁLCULO DE PROBABILIDADE DE COMPRA POR DESTINO
# ------------------------------------------------------------
# Soma tickets por cliente e destino
tickets_cliente = df_filtrado_por_ano.groupby(['ID_CLIENTE_SIMPLIFICADO', 'LOCAL_DESTINO_IDA_SIMPLIFICADO'])[
    'QUANTIDADE_TICKETS'].sum().reset_index()
tickets_cliente.rename(columns={'QUANTIDADE_TICKETS': 'TOTAL_TICKETS'}, inplace=True)

# Soma a quantidade de tickets por destino
volume_tickets_destino = tickets_cliente.groupby('LOCAL_DESTINO_IDA_SIMPLIFICADO')['TOTAL_TICKETS'].sum().reset_index()
volume_tickets_destino.rename(columns={'TOTAL_TICKETS': 'TOTAL_TICKETS_DESTINO'}, inplace=True)

# Soma a quantidade de tickets total da plataforma
total_tickets_geral = volume_tickets_destino['TOTAL_TICKETS_DESTINO'].sum()

# Calcula a probabilidade de compra por destino
probabilidade_destino = volume_tickets_destino.copy()
probabilidade_destino['PROBABILIDADE_DESTINO'] = probabilidade_destino['TOTAL_TICKETS_DESTINO'] / total_tickets_geral
probabilidade_destino = probabilidade_destino.sort_values(by='PROBABILIDADE_DESTINO', ascending=False)

# ------------------------------------------------------------
# CÁLCULO DA MÉDIA DE INTERVALO GLOBAL DA PLATAFORMA (FALLBACK)
# ------------------------------------------------------------
# Seleciona apenas as colunas necessárias do DataFrame principal para otimizar a memória
df_intervalos_globais = df_filtrado_por_ano[['ID_CLIENTE_SIMPLIFICADO', 'DATA_COMPRA']].copy()

# Ordena os dados por cliente e pela data da compra
df_intervalos_globais.sort_values(by=['ID_CLIENTE_SIMPLIFICADO', 'DATA_COMPRA'], inplace=True)

# Agrupa por cliente e calcula a diferença em dias entre cada compra consecutiva
df_intervalos_globais['INTERVALO_DIAS'] = df_intervalos_globais.groupby('ID_CLIENTE_SIMPLIFICADO')['DATA_COMPRA'].diff().dt.days

# Calcula a média de todos os valores na coluna 'INTERVALO_DIAS'.
media_global_plataforma = df_intervalos_globais['INTERVALO_DIAS'].mean()

# Verificação de segurança final
if pd.isna(media_global_plataforma):
    media_global_plataforma = 90.0

# ------------------------------------------------------------
# PREVISÃO DE DATAS DE COMPRA PARA CLIENTES DE ALTO VALOR (OU SEJA, CLIENTES QUE GASTAM ACIMA DA MÉDIA NA PLATAFORMA)
# ------------------------------------------------------------
# Calcula gasto médio por cliente
df_filtrado_por_ano['VALOR_TOTAL_COMPRA'] = df_filtrado_por_ano['VALOR_TICKET'] * df_filtrado_por_ano['QUANTIDADE_TICKETS']
gasto_por_cliente = df_filtrado_por_ano.groupby('ID_CLIENTE_SIMPLIFICADO')['VALOR_TOTAL_COMPRA'].sum().reset_index()
gasto_medio_total = gasto_por_cliente['VALOR_TOTAL_COMPRA'].mean()

# Filtra clientes que gastam acima da média na plataforma
clientes_acima_media = gasto_por_cliente[gasto_por_cliente['VALOR_TOTAL_COMPRA'] > gasto_medio_total]['ID_CLIENTE_SIMPLIFICADO'].tolist()
df_clientes_alto_valor = df_filtrado_por_ano[df_filtrado_por_ano['ID_CLIENTE_SIMPLIFICADO'].isin(clientes_acima_media)].copy()

# Calcula intervalo de dias entre as compras de clientes alto valor
df_clientes_alto_valor.sort_values(by=['ID_CLIENTE_SIMPLIFICADO', 'DATA_COMPRA'], inplace=True)
df_clientes_alto_valor['INTERVALO_DIAS'] = df_clientes_alto_valor.groupby('ID_CLIENTE_SIMPLIFICADO')['DATA_COMPRA'].diff().dt.days

# Calcula a média de intervalo de compras de clientes alto valor
media_intervalo = df_clientes_alto_valor.groupby('ID_CLIENTE_SIMPLIFICADO')['INTERVALO_DIAS'].mean().reset_index()
media_intervalo.rename(columns={'INTERVALO_DIAS': 'MEDIA_INTERVALO_DIAS'}, inplace=True)

# Seleciona data da última compra de clientes alto valor
ultima_compra = df_clientes_alto_valor.groupby('ID_CLIENTE_SIMPLIFICADO')['DATA_COMPRA'].max().reset_index()

# Faz o merge entre dataframes
previsao = ultima_compra.merge(media_intervalo, on='ID_CLIENTE_SIMPLIFICADO', how='left')

# --- CORREÇÃO DE DIVISÃO POR ZERO ---
# Trata o caso de intervalo médio ser 0 (transforma o 0 em NaN, forçando o uso da média geral para esse cliente)
previsao['MEDIA_INTERVALO_DIAS'].replace(0, np.nan, inplace=True)

# Lógica de preenchimento com fallback
media_geral_alto_valor = df_clientes_alto_valor['INTERVALO_DIAS'].mean()
media_a_ser_usada = media_geral_alto_valor if pd.notna(media_geral_alto_valor) else media_global_plataforma
previsao['MEDIA_INTERVALO_DIAS'] = previsao['MEDIA_INTERVALO_DIAS'].fillna(media_a_ser_usada)

# Variável que recebe a data de hoje
hoje = pd.Timestamp.today().normalize()

# Cacula a próxima compra (protegendo contra divisão por zero)
dias_passados = (hoje - previsao['DATA_COMPRA']).dt.days.clip(lower=0)
multiplicador = np.ceil(dias_passados / previsao['MEDIA_INTERVALO_DIAS']).fillna(1)
previsao['PROXIMA_COMPRA'] = previsao['DATA_COMPRA'] + pd.to_timedelta(
    multiplicador * previsao['MEDIA_INTERVALO_DIAS'], unit="D"
)
previsao['PROXIMA_COMPRA'] = previsao['PROXIMA_COMPRA'].dt.date

# Exibição
previsao_df_display = previsao.copy()
previsao_df_display['ID do Cliente'] = previsao_df_display['ID_CLIENTE_SIMPLIFICADO']
previsao_df_display['ÚLTIMA COMPRA'] = previsao_df_display['DATA_COMPRA'].dt.strftime('%d/%m/%Y')
previsao_df_display['PRÓXIMA COMPRA PREVISTA'] = pd.to_datetime(
    previsao_df_display['PROXIMA_COMPRA'], errors='coerce'
).dt.strftime('%d/%m/%Y')

# ------------------------------------------------------------
# LÓGICA DE SEGMENTAÇÃO DE CLIENTES
# ------------------------------------------------------------
# Seleciona data de última compra e considera as compras somente nos últimos 3 anos
data_maxima = df_filtrado_por_ano['DATA_COMPRA'].max().date()
data_limite_atividade = pd.Timestamp(data_maxima) - pd.DateOffset(years=3)

# Agrupa por cliente as últimas datas de compra
contagem_ultima_compra = df_filtrado_por_ano.groupby('ID_CLIENTE_SIMPLIFICADO')['DATA_COMPRA'].max().reset_index()

# Conta clientes ativos (compraram nos últimos 3 anos) e inativos (não compraram nos últimos 3 anos)
clientes_ativos = contagem_ultima_compra[contagem_ultima_compra['DATA_COMPRA'] >= data_limite_atividade]['ID_CLIENTE_SIMPLIFICADO'].nunique()
clientes_inativos = df_filtrado_por_ano['ID_CLIENTE_SIMPLIFICADO'].nunique() - clientes_ativos

# Cria a lista de clientes frequentes (clientes que fizeram mais de uma compra na plataforma)
contagem_compras = df_filtrado_por_ano.groupby('ID_CLIENTE_SIMPLIFICADO').size().reset_index(name='NUM_COMPRAS')
clientes_frequentes = contagem_compras[contagem_compras['NUM_COMPRAS'] > 1]['ID_CLIENTE_SIMPLIFICADO'].tolist()

# Cria a lista de "Clientes Vip" (clientes que gastam acima da média e compraram mais de uma vez na plataforma)
clientes_alto_valor_e_frequencia = list(set(clientes_acima_media) & set(clientes_frequentes))

# Cria dataframe com as diferentes categorias de clientes
df_segmentacao = pd.DataFrame({
    'Categoria': [
        'Clientes de Alto Valor',
        'Clientes Vip',
        'Clientes Ativos',
        'Clientes Inativos'
    ],
    'Quantidade': [
        len(clientes_acima_media),
        len(clientes_alto_valor_e_frequencia),
        clientes_ativos,
        clientes_inativos
    ]
})

# ------------------------------------------------------------
# CONSTRUÇÃO DOS GRÁFICOS
# ------------------------------------------------------------
st.markdown("---")
col_prob, col_seg_baixo = st.columns(2)

with col_prob:
    st.header('Próximos Destinos de Compra')
    top_20_probabilidade = probabilidade_destino.head(20)
    fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
    ax_prob.bar(
        top_20_probabilidade['LOCAL_DESTINO_IDA_SIMPLIFICADO'].astype(str),
        top_20_probabilidade['PROBABILIDADE_DESTINO'],
        color='#55ACCF'
    )
    ax_prob.yaxis.set_major_formatter(PercentFormatter(1))
    ax_prob.set_xlabel('ID do Destino', fontsize=10)
    ax_prob.set_ylabel('Probabilidade Média', fontsize=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_prob)

    with col_seg_baixo:
        st.header('Segmentação de Clientes')
        cores_segmentacao = ['#115DCF', '#55ACCF', '#11CFC8', '#ED452B']
        chart_segmentacao_baixo = alt.Chart(df_segmentacao).mark_bar().encode(
            x=alt.X('Categoria:N', title='Segmento', sort=None,
                    axis=alt.Axis(labels=True, labelAngle=0, labelLimit=200, labelFontSize=10)),
            y=alt.Y('Quantidade:Q', title='Quantidade de Clientes'),
            color=alt.Color('Categoria:N', scale=alt.Scale(range=cores_segmentacao),
                            legend=alt.Legend(title="Categorias", orient="bottom")),
            tooltip=['Categoria', 'Quantidade']
        ).interactive()
        st.altair_chart(chart_segmentacao_baixo, use_container_width=True)

st.markdown("---")
col_5dias, col_prev = st.columns(2)

with col_5dias:
    st.header('Previsão para os Próximos 5 Dias')

    data_inicio_filtro = datetime.date(2025, 9, 14)
    data_fim_filtro = data_inicio_filtro + datetime.timedelta(days=4)

    previsao_5_dias = previsao[
        (pd.to_datetime(previsao['PROXIMA_COMPRA']).dt.date >= data_inicio_filtro) &
        (pd.to_datetime(previsao['PROXIMA_COMPRA']).dt.date <= data_fim_filtro)
    ].copy()

    if previsao_5_dias.empty:
        st.info(
            f"Nenhuma previsão de compra encontrada para o período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}."
        )
    else:
        previsao_5_dias_com_destino = pd.merge(
            previsao_5_dias,
            df_filtrado_por_ano[['ID_CLIENTE_SIMPLIFICADO', 'LOCAL_DESTINO_IDA_SIMPLIFICADO']].drop_duplicates(),
            on='ID_CLIENTE_SIMPLIFICADO',
            how='left'
        )

        previsao_5_dias_com_destino['DESTINO_IDA'] = previsao_5_dias_com_destino.groupby('ID_CLIENTE_SIMPLIFICADO')[
            'LOCAL_DESTINO_IDA_SIMPLIFICADO'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)

        previsao_5_dias_com_destino.drop_duplicates(subset='ID_CLIENTE_SIMPLIFICADO', inplace=True)
        previsao_5_dias_com_destino.dropna(subset=['DESTINO_IDA'], inplace=True)

        df_chart_final = previsao_5_dias_com_destino.groupby(
            [pd.to_datetime(previsao_5_dias_com_destino['PROXIMA_COMPRA']).dt.strftime('%d/%m'), 'DESTINO_IDA']
        ).size().reset_index(name='Quantidade de Clientes')

        df_chart_final.rename(columns={'PROXIMA_COMPRA': 'Data'}, inplace=True)

        cores_personalizadas = [
            '#1092C6', '#0D79A4', '#1199CF', '#0C6CBF', '#0A6082',
            '#084760', '#13303D', '#287BE0', '#5984D6', '#5689F5',
            '#115FD4', '#0F56BF', '#115DCF', '#09316E', '#0E4DAB',
            '#0C4396', '#0A3A82', '#072B59', '#061F45', '#1471FC',
            '#1048C2', '#0E40AD'
        ]

        destinos_unicos = sorted(df_chart_final['DESTINO_IDA'].unique())
        escala_cores = alt.Scale(domain=destinos_unicos, range=cores_personalizadas[:len(destinos_unicos)])

        chart = alt.Chart(df_chart_final).mark_bar().encode(
            x=alt.X('Data:O', axis=alt.Axis(title="Data prevista")),
            y=alt.Y('Quantidade de Clientes:Q', title='Clientes com previsão de compra'),
            color=alt.Color('DESTINO_IDA:N', scale=escala_cores, title='Destino', legend=alt.Legend(orient="bottom")),
            tooltip=['Data:N', 'DESTINO_IDA:N', 'Quantidade de Clientes:Q']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

with col_prev:
    st.header('Previsão de Próxima Compra')
    st.dataframe(previsao_df_display[['ID do Cliente', 'ÚLTIMA COMPRA', 'PRÓXIMA COMPRA PREVISTA']], hide_index=True)
 