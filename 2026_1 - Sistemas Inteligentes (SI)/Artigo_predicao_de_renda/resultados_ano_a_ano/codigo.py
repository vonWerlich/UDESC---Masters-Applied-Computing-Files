import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import TwoSlopeNorm

# Criar pasta para os resultados convencionais
os.makedirs('graficos_completos_convencional', exist_ok=True)

# ----------------------------
# 1. Ler dados (Convencional)
# ----------------------------
df_ajust_log = pd.read_csv("resultados_regressao_ajustado_log.csv")
df_ajust_log['Config'] = 'Ajustado + Log'

df_ajust_norm = pd.read_csv("resultados_regressao_ajustado_normal.csv")
df_ajust_norm['Config'] = 'Ajustado + Normal'

df_nom_log = pd.read_csv("resultados_regressao_nominal_log.csv")
df_nom_log['Config'] = 'Nominal + Log'

df_nom_norm = pd.read_csv("resultados_regressao_nominal_normal.csv")
df_nom_norm['Config'] = 'Nominal + Normal'

df_all = pd.concat(
    [df_ajust_log, df_ajust_norm, df_nom_log, df_nom_norm],
    ignore_index=True
)

# ----------------------------
# 2. Métricas
# ----------------------------
colunas_ignorar = ['Ano_Treino', 'Ano_Teste', 'Distancia_Anos', 'Config']
metricas = [c for c in df_all.columns if c not in colunas_ignorar]

# ----------------------------
# 3. Heatmap
# ----------------------------
def plot_matriz_temporal(metric):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    configs = ['Ajustado + Log', 'Ajustado + Normal', 'Nominal + Log', 'Nominal + Normal']

    is_divergent = False
    if 'R2' in metric:
        cmap = 'viridis'
    elif 'Disp' in metric or ('ME' in metric and 'RMSE' not in metric and 'MAE' not in metric and 'MAPE' not in metric):
        cmap = 'RdBu_r'
        is_divergent = True
    else:
        cmap = 'YlOrRd'

    vmin = df_all[metric].min()
    vmax = df_all[metric].max()

    norm = None
    if is_divergent and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    hm = None
    for i, config in enumerate(configs):
        df_sub = df_all[df_all['Config'] == config]
        
        # A função pivot agrupa automaticamente: 
        # Diagonal (Ano_Treino == Ano_Teste) é o baseline
        # Triângulo superior (Ano_Treino < Ano_Teste) são as projeções
        pivot = df_sub.pivot(index='Ano_Treino', columns='Ano_Teste', values=metric)
        
        fmt_str = ".3f" if any(x in metric for x in ["R2", "MAPE"]) else ".0f"

        hm = sns.heatmap(
            pivot, annot=True, fmt=fmt_str, cmap=cmap, norm=norm,
            vmin=None if norm else vmin, vmax=None if norm else vmax,
            linewidths=0.5, linecolor='white', cbar=False,
            annot_kws={"size": 9, "weight": "bold"}, ax=axes[i]
        )

        axes[i].set_title(config, fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Ano Treino', fontsize=11)
        axes[i].set_xlabel('Ano Teste', fontsize=11)
        axes[i].tick_params(axis='x', rotation=45)

    cbar = fig.colorbar(hm.collections[0], ax=axes, shrink=0.92, aspect=35, pad=0.02)
    cbar.set_label(metric, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle(f'Matriz Temporal (Convencional) - {metric}', fontsize=17, fontweight='bold')
    filename = f"graficos_completos_convencional/heatmap_{metric.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {filename}")

print(f"A gerar {len(metricas)} gráficos para a Abordagem Convencional...")
for metrica in metricas:
    plot_matriz_temporal(metrica)
print("Concluído.")