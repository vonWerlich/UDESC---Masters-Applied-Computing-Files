from folktables import ACSDataSource, BasicProblem, adult_filter
import pandas as pd
import numpy as np
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==========================================
# 1. A MÁGICA DA REGRESSÃO: Criando a nossa própria tarefa
# ==========================================
# Em vez de importar o ACSIncome engessado, nós definimos o nosso próprio problema.
# Note que a propriedade 'target_transform' foi removida, o que significa que
# o alvo (target) continuará sendo o valor monetário bruto (PINCP).
ACS_Income_Regression = BasicProblem(
    features=[
        'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP',
        'WKHP', 'SEX', 'RAC1P',
    ],
    target='PINCP', 
    target_transform=None, # O segredo está aqui: nenhuma transformação para True/False
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, nan=-1.0),
)

# ==========================================
# 2. Configurações de Download
# ==========================================
anos_pesquisa = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
estados_mini_eua = ["IL", "PA", "NC", "CO"] 

lista_dataframes = []

for ano in anos_pesquisa:
    print(f"Estruturando microdados do Censo para o ano {ano}...")
    
    data_source = ACSDataSource(survey_year=str(ano), horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=estados_mini_eua, download=True)
    
    if 'RELSHIPP' in acs_data.columns and 'RELP' not in acs_data.columns:
        acs_data = acs_data.rename(columns={'RELSHIPP': 'RELP'})
    
    # Usamos o nosso problema customizado. 
    # Ele mesmo já aplica o adult_filter internamente (definido no preprocess acima)
    features, label, group = ACS_Income_Regression.df_to_pandas(acs_data)
    
    df_ano = features.copy()
    
    # Como não usamos target_transform, 'label' agora carrega dólares, não booleanos!
    df_ano['RENDA_CONTINUA'] = label 
    df_ano['RACA_SENSIVEL'] = group
    df_ano['ANO_REFERENCIA'] = ano
    
    lista_dataframes.append(df_ano)

# ==========================================
# 3. Fechamento e Exportação
# ==========================================
df_completo = pd.concat(lista_dataframes, ignore_index=True)

df_completo.to_csv("Base_Folktables_Regressao.csv", index=False)

print("\n=== Estruturação Concluída ===")
print(f"Total de registros na base: {len(df_completo)} linhas.")