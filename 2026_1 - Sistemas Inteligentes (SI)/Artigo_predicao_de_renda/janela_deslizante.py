import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURAÇÕES LOCAIS
# ==========================================
ANOS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
N_RODADAS = 30 
N_SPLITS_KFOLD = 5 

CENARIOS = [
    {"tipo": "nominal", "pasta": "./dados_nominais"},
    {"tipo": "ajustado", "pasta": "./dados_ajustados"}
]

TRANSFORMACOES = ["normal", "log"]
PASTA_RESULTADOS = './resultados_experimentos'
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# ==========================================
# 2. CARREGAMENTO EM MEMÓRIA (CACHE DE INICIAÇÃO)
# ==========================================
print("[*] Carregando todos os DataFrames em memória para as janelas deslizantes...")
cache_dados = {}
for cenario in CENARIOS:
    tipo = cenario["tipo"]
    pasta = cenario["pasta"]
    cache_dados[tipo] = {}
    for ano in ANOS:
        caminho_arquivo = f"{pasta}/acs_{ano}_{tipo}.csv"
        if os.path.exists(caminho_arquivo):
            cache_dados[tipo][ano] = pd.read_csv(caminho_arquivo)
print("[*] Todos os dados foram cacheados com sucesso!\n")

# ==========================================
# 3. METRICAS E AUDITORIA
# ==========================================
def calcular_todas_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, me, mape, rmse, r2

def auditoria_regressao(y_true, y_pred, coluna_sensivel, valor_privilegiado):
    y_t, y_p, sens = np.array(y_true), np.array(y_pred), np.array(coluna_sensivel)
    mask_p = (sens == valor_privilegiado)
    mask_u = (sens != valor_privilegiado)
    
    if sum(mask_p) == 0 or sum(mask_u) == 0:
        return [np.nan] * 9
        
    mae_p, me_p, mape_p, _, _ = calcular_todas_metricas(y_t[mask_p], y_p[mask_p])
    mae_u, me_u, mape_u, _, _ = calcular_todas_metricas(y_t[mask_u], y_p[mask_u])
    
    disp_mae = abs(mae_p - mae_u)
    disp_me = abs(me_p - me_u)
    disp_mape = abs(mape_p - mape_u)
    
    return [mae_p, mae_u, disp_mae, me_p, me_u, disp_me, mape_p, mape_u, disp_mape]

# ==========================================
# 4. LOOP DE TREINAMENTO (JANELAS DESLIZANTES COM BASELINE)
# ==========================================
COLUNAS_CSV_JANELA = [
    'Ano_Treino_1', 'Ano_Treino_2', 'Ano_Teste',
    'MAE_Geral', 'ME_Geral', 'MAPE_Geral', 'RMSE_Geral', 'R2_Geral',
    'Raca_MAE_Priv', 'Raca_MAE_Unpriv', 'Disp_MAE_Raca',
    'Raca_ME_Priv', 'Raca_ME_Unpriv', 'Disp_ME_Raca',
    'Raca_MAPE_Priv', 'Raca_MAPE_Unpriv', 'Disp_MAPE_Raca',
    'Gen_MAE_Priv', 'Gen_MAE_Unpriv', 'Disp_MAE_Gen',
    'Gen_ME_Priv', 'Gen_ME_Unpriv', 'Disp_ME_Gen',
    'Gen_MAPE_Priv', 'Gen_MAPE_Unpriv', 'Disp_MAPE_Gen'
]

print("\n=== FASE 3: EXPERIMENTOS DE REGRESSÃO COM JANELAS DESLIZANTES (2 ANOS) E BASELINE ===")

for cenario in CENARIOS:
    tipo = cenario["tipo"]
    
    for transformacao in TRANSFORMACOES:
        caminho_csv_janela = f"{PASTA_RESULTADOS}/resultados_janela_{tipo}_{transformacao}.csv"
        
        combinacoes_feitas_janela = set()
        if os.path.exists(caminho_csv_janela):
            df_ex_janela = pd.read_csv(caminho_csv_janela)
            # A chave única agora é a trinca: (Ano1, Ano2, Ano_Teste)
            combinacoes_feitas_janela = set(zip(df_ex_janela['Ano_Treino_1'], df_ex_janela['Ano_Treino_2'], df_ex_janela['Ano_Teste']))
            print(f"\n[*] Retomando JANELA {tipo.upper()} ({transformacao.upper()}). {len(combinacoes_feitas_janela)} já concluídas.")
        else:
            pd.DataFrame(columns=COLUNAS_CSV_JANELA).to_csv(caminho_csv_janela, index=False)
            print(f"\n[*] Iniciando JANELA {tipo.upper()} ({transformacao.upper()}) do zero.")

        # Iterar garantindo que temos pelo menos 2 anos para treino e 1 para teste
        for i in range(len(ANOS) - 2):
            ano_t1 = ANOS[i]
            ano_t2 = ANOS[i+1]
            ano_teste_futuro = ANOS[i+2]
            
            # Definimos que quando o Ano_Teste for igual ao Ano_Treino_2, representará o Baseline (K-Fold interno)
            testes_necessarios = [ano_t2, ano_teste_futuro]
            
            # Filtra apenas o que ainda não foi executado
            testes_pendentes = [t for t in testes_necessarios if (ano_t1, ano_t2, t) not in combinacoes_feitas_janela]
            
            if not testes_pendentes:
                continue
                
            print(f"\n>> Processando Janela: Treino [{ano_t1}, {ano_t2}] ({tipo} - {transformacao})")
            
            # Concatenar os DataFrames dos dois anos da janela temporal
            df_treino_1 = cache_dados[tipo][ano_t1]
            df_treino_2 = cache_dados[tipo][ano_t2]
            df_treino_concat = pd.concat([df_treino_1, df_treino_2], ignore_index=True)
            
            X_train = df_treino_concat.drop(columns=['TARGET', 'GROUP', 'YEAR'])
            y_train = df_treino_concat['TARGET']
            y_train_modelo = np.log1p(y_train) if transformacao == "log" else y_train
            
            # Dicionário para acumular resultados do Baseline e do Futuro
            acumulador_janela = {ano: {col: [] for col in COLUNAS_CSV_JANELA[3:]} for ano in testes_pendentes}
            
            for semente in range(N_RODADAS):
                
                # ---------------------------------------------------------
                # ETAPA A: BASELINE K-FOLD (O "TESTE" NA PRÓPRIA JANELA)
                # ---------------------------------------------------------
                if ano_t2 in testes_pendentes:
                    kf = KFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=semente)
                    folds_acc = {col: [] for col in COLUNAS_CSV_JANELA[3:]}
                    
                    X_np, y_mod_np, y_real_np = X_train.values, y_train_modelo.values, y_train.values
                    raca_np, sexo_np = X_train['RAC1P'].values, X_train['SEX'].values
                    
                    for train_idx, val_idx in kf.split(X_np):
                        X_kf_train, y_kf_train = X_np[train_idx], y_mod_np[train_idx]
                        X_kf_val, y_kf_val_real = X_np[val_idx], y_real_np[val_idx]
                        
                        rf_cv = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=semente, n_jobs=-1)
                        rf_cv.fit(X_kf_train, y_kf_train)
                        
                        y_pred_modelo = rf_cv.predict(X_kf_val)
                        y_pred_val = np.expm1(y_pred_modelo) if transformacao == "log" else y_pred_modelo
                        
                        mae, me, mape, rmse, r2 = calcular_todas_metricas(y_kf_val_real, y_pred_val)
                        met_raca = auditoria_regressao(y_kf_val_real, y_pred_val, raca_np[val_idx], 1)
                        met_sexo = auditoria_regressao(y_kf_val_real, y_pred_val, sexo_np[val_idx], 1)
                        
                        linha_fold = [mae, me, mape, rmse, r2] + met_raca + met_sexo
                        for chave, valor in zip(folds_acc.keys(), linha_fold):
                            folds_acc[chave].append(valor)
                            
                    for col in COLUNAS_CSV_JANELA[3:]:
                        acumulador_janela[ano_t2][col].append(np.mean(folds_acc[col]))

                # ---------------------------------------------------------
                # ETAPA B: PROJEÇÃO FUTURA (TREINO NA JANELA COMPLETA)
                # ---------------------------------------------------------
                if ano_teste_futuro in testes_pendentes:
                    rf_janela = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=semente, n_jobs=-1)
                    rf_janela.fit(X_train, y_train_modelo)
                    
                    df_teste = cache_dados[tipo][ano_teste_futuro]
                    X_test = df_teste.drop(columns=['TARGET', 'GROUP', 'YEAR'])
                    y_test = df_teste['TARGET']
                    
                    y_pred_modelo = rf_janela.predict(X_test)
                    y_pred = np.expm1(y_pred_modelo) if transformacao == "log" else y_pred_modelo
                        
                    mae, me, mape, rmse, r2 = calcular_todas_metricas(y_test, y_pred)
                    met_raca = auditoria_regressao(y_test, y_pred, X_test['RAC1P'], 1)
                    met_sexo = auditoria_regressao(y_test, y_pred, X_test['SEX'], 1)
                    
                    linha_resultado = [mae, me, mape, rmse, r2] + met_raca + met_sexo
                    for chave_col, valor in zip(acumulador_janela[ano_teste_futuro].keys(), linha_resultado):
                        acumulador_janela[ano_teste_futuro][chave_col].append(valor)
            
            # Consolidar a média das rodadas e salvar no CSV
            for t in testes_pendentes:
                media_final_janela = {
                    'Ano_Treino_1': ano_t1, 
                    'Ano_Treino_2': ano_t2, 
                    'Ano_Teste': t
                }
                
                for col in COLUNAS_CSV_JANELA[3:]:
                    media_final_janela[col] = np.mean(acumulador_janela[t][col])
                    
                pd.DataFrame([media_final_janela]).to_csv(caminho_csv_janela, mode='a', header=False, index=False)
                tag = "BASELINE" if t == ano_t2 else "FUTURO  "
                print(f"   [{tag}] Teste {t} | MAE Geral: ${media_final_janela['MAE_Geral']:.0f} | Disp. MAE Raça: ${media_final_janela['Disp_MAE_Raca']:.0f}")

print("\n[*] Todos os experimentos com janelas deslizantes foram concluídos!")