import os
import pandas as pd
import numpy as np
from folktables import ACSDataSource, BasicProblem, adult_filter

# 1. Configurações Iniciais do Experimento
ANOS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
ESTADOS = ["IL", "PA", "NC", "CO"]

# Valores brutos do CPI-U (Média Anual) - U.S. Bureau of Labor Statistics
# Fonte: https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202402.pdf
CPI_BRUTO_BLS = {
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2021: 270.970,
    2022: 292.655
}

# Define o ano-base para fixar o poder de compra (Fidelidade a 2014)
ANO_BASE_INFLACAO = 2014
CPI_REFERENCIA = CPI_BRUTO_BLS[ANO_BASE_INFLACAO]

# Criar diretórios de saída
os.makedirs("dados_nominais", exist_ok=True)
os.makedirs("dados_ajustados", exist_ok=True)

# ==========================================
# 2. Definição da Tarefa de Regressão (A Nova Lógica)
# ==========================================
ACS_Income_Regression = BasicProblem(
    features=[
        'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP',
        'WKHP', 'SEX', 'RAC1P',
    ],
    target='PINCP', # O salário exato será o alvo
    target_transform=None, # Sem transformação para True/False
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, nan=-1.0),
)

for ano in ANOS:
    print(f"\n--- Processando Ano: {ano} ---")
    
    # Download/Carga dos dados brutos
    data_source = ACSDataSource(survey_year=str(ano), horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=ESTADOS, download=True)
    
    # Tratamento de consistência do Censo (RELP vs RELSHIPP)
    if 'RELSHIPP' in acs_data.columns:
        acs_data = acs_data.rename(columns={'RELSHIPP': 'RELP'})
    
    # Extração de Atributos usando nosso problema de Regressão customizado
    # O preprocess (adult_filter) já é aplicado automaticamente pelo df_to_pandas
    features, label, group = ACS_Income_Regression.df_to_pandas(acs_data)
    
    # --- VERSÃO 1: DADOS NOMINAIS (Regressão: Valores Reais em Dólares Correntes) ---
    df_nominal = features.copy()
    # Como não tem target_transform, 'label' já contém a renda contínua
    df_nominal['TARGET'] = label.values 
    df_nominal['GROUP'] = group.values
    df_nominal['YEAR'] = ano
    
    path_nom = f"dados_nominais/acs_{ano}_nominal.csv"
    df_nominal.to_csv(path_nom, index=False)
    print(f"Salvo: {path_nom} (Sem correção)")
    
    # --- VERSÃO 2: DADOS AJUSTADOS (Correção Inflacionária Dinâmica para Regressão) ---
    df_ajustado = features.copy()
    
    # CÁLCULO DINÂMICO: (CPI de 2014 / CPI do Ano Corrente)
    multiplicador_deflacao = CPI_REFERENCIA / CPI_BRUTO_BLS[ano]
    
    # Aplica a correção matemática: Apenas multiplica, sem checar se é > 50000
    df_ajustado['TARGET'] = label.values * multiplicador_deflacao
    df_ajustado['GROUP'] = group.values
    df_ajustado['YEAR'] = ano
    
    path_aju = f"dados_ajustados/acs_{ano}_ajustado.csv"
    df_ajustado.to_csv(path_aju, index=False)
    print(f"Salvo: {path_aju} (Multiplicador aplicado: {multiplicador_deflacao:.4f})")

print("\n=== Preparação da Matriz Temporal (Regressão) Concluída! ===")