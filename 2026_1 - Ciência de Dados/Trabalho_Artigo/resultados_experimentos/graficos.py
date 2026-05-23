import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Criar uma pasta para guardar os gráficos, evitando desorganizar a diretoria raiz
os.makedirs('graficos_completos', exist_ok=True)

# 1. Carregar e identificar os datasets
df_ajust_log = pd.read_csv("resultados_regressao_ajustado_log.csv")
df_ajust_log['Config'] = 'Ajustado + Log'

df_ajust_norm = pd.read_csv("resultados_regressao_ajustado_normal.csv")
df_ajust_norm['Config'] = 'Ajustado + Normal'

df_nom_log = pd.read_csv("resultados_regressao_nominal_log.csv")
df_nom_log['Config'] = 'Nominal + Log'

df_nom_norm = pd.read_csv("resultados_regressao_nominal_normal.csv")
df_nom_norm['Config'] = 'Nominal + Normal'

df_all = pd.concat([df_ajust_log, df_ajust_norm, df_nom_log, df_nom_norm], ignore_index=True)

# 2. Identificar automaticamente todas as colunas de métricas
colunas_ignorar = ['Ano_Treino', 'Ano_Teste', 'Distancia_Anos', 'Config']
metricas = [col for col in df_all.columns if col not in colunas_ignorar]

# 3. Função para gerar e guardar a matriz de cada métrica
def plot_matriz_temporal(metric):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    configs = ['Ajustado + Log', 'Ajustado + Normal', 'Nominal + Log', 'Nominal + Normal']
    
    # Lógica de seleção de cores baseada no nome da métrica
    is_divergent = False
    if 'R2' in metric:
        cmap = 'mako'
    elif 'Disp' in metric or ('ME' in metric and 'RMSE' not in metric and 'MAE' not in metric and 'MAPE' not in metric):
        cmap = 'coolwarm'
        is_divergent = True
    else:
        cmap = 'rocket_r'

    # Limites globais para manter a comparabilidade das cores
    vmin = df_all[metric].min()
    vmax = df_all[metric].max()
    
    # Se for uma métrica divergente (como ME ou Disparidade), ajuda muito se o "zero" for a cor neutra
    center = 0 if is_divergent and vmin < 0 < vmax else None

    for i, config in enumerate(configs):
        df_sub = df_all[df_all['Config'] == config]
        pivot = df_sub.pivot(index='Ano_Treino', columns='Ano_Teste', values=metric)
        
        # Formatação: se for métrica percentual/fracional (R2, MAPE), usar 3 casas decimais. Senão, números inteiros.
        fmt_str = ".3f" if any(x in metric for x in ["R2", "MAPE"]) else ".0f"
        
        # Desenhar o heatmap
        if center is not None:
            sns.heatmap(pivot, annot=True, fmt=fmt_str, cmap=cmap, ax=axes[i], 
                        vmin=vmin, vmax=vmax, center=center, cbar_kws={'label': metric}, linewidths=.5)
        else:
            sns.heatmap(pivot, annot=True, fmt=fmt_str, cmap=cmap, ax=axes[i], 
                        vmin=vmin, vmax=vmax, cbar_kws={'label': metric}, linewidths=.5)
        
        axes[i].set_title(config, fontsize=14)
        axes[i].set_ylabel('Ano de Treinamento')
        axes[i].set_xlabel('Ano de Teste')

    plt.suptitle(f'Matriz de {metric} (Treino x Teste)', fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Guardar a imagem na pasta dedicada
    filename = f"graficos_completos/heatmap_{metric.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado com sucesso: {filename}")

# 4. Executar o loop para todas as métricas encontradas
print(f"A gerar gráficos para {len(metricas)} métricas. Aguarde um momento...")
for metrica in metricas:
    plot_matriz_temporal(metrica)

print("\nProcesso concluído! Verifique a pasta 'graficos_completos'.")