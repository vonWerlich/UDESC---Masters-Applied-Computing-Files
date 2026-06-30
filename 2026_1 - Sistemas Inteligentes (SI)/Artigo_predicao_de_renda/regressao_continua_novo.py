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
N_RODADAS = 30 # Mantenha 30 se tiver tempo computacional, ou reduza para 5/10 para testes rápidos
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
print("[*] Carregando todos os DataFrames em memória para otimizar velocidade...")
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
# 4. LOOP DE TREINAMENTO (COM BASELINE VIA K-FOLD)
# ==========================================
COLUNAS_CSV = [
    'Ano_Treino', 'Ano_Teste', 'Distancia_Anos',
    'MAE_Geral', 'ME_Geral', 'MAPE_Geral', 'RMSE_Geral', 'R2_Geral',
    'Raca_MAE_Priv', 'Raca_MAE_Unpriv', 'Disp_MAE_Raca',
    'Raca_ME_Priv', 'Raca_ME_Unpriv', 'Disp_ME_Raca',
    'Raca_MAPE_Priv', 'Raca_MAPE_Unpriv', 'Disp_MAPE_Raca',
    'Gen_MAE_Priv', 'Gen_MAE_Unpriv', 'Disp_MAE_Gen',
    'Gen_ME_Priv', 'Gen_ME_Unpriv', 'Disp_ME_Gen',
    'Gen_MAPE_Priv', 'Gen_MAPE_Unpriv', 'Disp_MAPE_Gen'
]

print("=== FASE 2: EXPERIMENTOS DE REGRESSÃO COM BASELINE OTIMIZADO ===")

for cenario in CENARIOS:
    tipo = cenario["tipo"]
    
    for transformacao in TRANSFORMACOES:
        caminho_csv = f"{PASTA_RESULTADOS}/resultados_regressao_{tipo}_{transformacao}.csv"
        
        combinacoes_feitas = set()
        if os.path.exists(caminho_csv):
            df_ex = pd.read_csv(caminho_csv)
            combinacoes_feitas = set(zip(df_ex['Ano_Treino'], df_ex['Ano_Teste']))
            print(f"\n[*] Retomando {tipo.upper()} ({transformacao.upper()}). {len(combinacoes_feitas)} já concluídas.")
        else:
            pd.DataFrame(columns=COLUNAS_CSV).to_csv(caminho_csv, index=False)
            print(f"\n[*] Iniciando {tipo.upper()} ({transformacao.upper()}) do zero.")

        # Iteramos sobre todos os anos. O último ano servirá apenas para gerar o baseline (Distância 0)
        for ano_treino in ANOS:
            anos_teste = [ano for ano in ANOS if ano > ano_treino]
            
            # Adiciona o próprio ano de treino como o primeiro "teste" (Baseline)
            todos_testes_necessarios = [ano_treino] + anos_teste
            
            # Filtra os testes pendentes
            testes_pendentes = [ano for ano in todos_testes_necessarios if (ano_treino, ano) not in combinacoes_feitas]
            if not testes_pendentes:
                continue
                
            print(f"\n>> Processando Bloco Temporal: Treino {ano_treino} ({tipo} - {transformacao})")
            
            df_treino = cache_dados[tipo][ano_treino]
            X_train = df_treino.drop(columns=['TARGET', 'GROUP', 'YEAR'])
            y_train = df_treino['TARGET']
            y_train_modelo = np.log1p(y_train) if transformacao == "log" else y_train
            
            acumulador_testes = {ano: {col: [] for col in COLUNAS_CSV[3:]} for ano in testes_pendentes}
            
            for semente in range(N_RODADAS):
                # ---------------------------------------------------------
                # ETAPA A: BASELINE K-FOLD (O "TESTE" COM O PRÓPRIO TREINO)
                # ---------------------------------------------------------
                if ano_treino in testes_pendentes:
                    kf = KFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=semente)
                    
                    # Variáveis temporárias para acumular os resultados dos Folds nesta semente
                    folds_acc = {col: [] for col in COLUNAS_CSV[3:]}
                    
                    # Convertendo para numpy/iloc para garantir compatibilidade de índices no KFold
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
                    
                    # Salva a média dos folds como o resultado do Baseline desta semente específica
                    for col in COLUNAS_CSV[3:]:
                        acumulador_testes[ano_treino][col].append(np.mean(folds_acc[col]))

                # ---------------------------------------------------------
                # ETAPA B: PROJEÇÃO FUTURA (TREINO COMPLETO)
                # ---------------------------------------------------------
                testes_futuros = [ano for ano in testes_pendentes if ano != ano_treino]
                if testes_futuros:
                    # Treina no dataset inteiro do ano para projetar no futuro
                    rf_full = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=semente, n_jobs=-1)
                    rf_full.fit(X_train, y_train_modelo)
                    
                    for ano_teste in testes_futuros:
                        df_teste = cache_dados[tipo][ano_teste]
                        X_test, y_test = df_teste.drop(columns=['TARGET', 'GROUP', 'YEAR']), df_teste['TARGET']
                        
                        y_pred_modelo = rf_full.predict(X_test)
                        y_pred = np.expm1(y_pred_modelo) if transformacao == "log" else y_pred_modelo
                            
                        mae, me, mape, rmse, r2 = calcular_todas_metricas(y_test, y_pred)
                        met_raca = auditoria_regressao(y_test, y_pred, X_test['RAC1P'], 1)
                        met_sexo = auditoria_regressao(y_test, y_pred, X_test['SEX'], 1)
                        
                        linha = [mae, me, mape, rmse, r2] + met_raca + met_sexo
                        for chave, valor in zip(acumulador_testes[ano_teste].keys(), linha):
                            acumulador_testes[ano_teste][chave].append(valor)
            
            # Salva os resultados no CSV
            for ano_teste in testes_pendentes:
                media_final = {'Ano_Treino': ano_treino, 'Ano_Teste': ano_teste, 'Distancia_Anos': ano_teste - ano_treino}
                for col in COLUNAS_CSV[3:]:
                    media_final[col] = np.mean(acumulador_testes[ano_teste][col])
                    
                pd.DataFrame([media_final]).to_csv(caminho_csv, mode='a', header=False, index=False)
                tag = "BASELINE" if ano_treino == ano_teste else "FUTURO  "
                print(f"   [{tag}] Teste {ano_teste} | MAE Geral: ${media_final['MAE_Geral']:.0f} | Disp. MAE Raça: ${media_final['Disp_MAE_Raca']:.0f}")