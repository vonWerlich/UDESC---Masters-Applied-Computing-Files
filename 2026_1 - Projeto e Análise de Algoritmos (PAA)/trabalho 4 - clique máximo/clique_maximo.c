#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXN 2048
#define WORDS ((MAXN + 63) / 64)
#define TIME_LIMIT_SECS 900.0 // Limite de 15 minutos por algoritmo
#define LINE_SIZE 8192

typedef unsigned long long U64;

static U64 adj[MAXN][WORDS];
static U64 old_adj[MAXN][WORDS];
static int n_vertices = 0;
static int n_edges = 0;

static int old_degree[MAXN];
static int original_label[MAXN];
static int perm[MAXN];

static unsigned short current_clique[MAXN];

// Estruturas para Força Bruta
static unsigned short best_clique_bf[MAXN];
static long long calls_bf = 0;
static int best_size_bf = 0;
static int timeout_bf = 0;
static double time_bf = 0.0;
static clock_t start_clock_bf;

// Estruturas para Backtracking
static unsigned short best_clique_bt[MAXN];
static long long calls_bt = 0;
static int best_size_bt = 0;
static int timeout_bt = 0;
static double time_bt = 0.0;
static clock_t start_clock_bt;

// Estruturas para Branch and Bound
static unsigned short best_clique_bb[MAXN];
static long long calls_bb = 0;
static long long prunes_bb = 0;
static int best_size_bb = 0;
static int timeout_bb = 0;
static double time_bb = 0.0;
static clock_t start_clock_bb;

/* ---------- helpers ---------- */

static int bit_index64(U64 x) {
#if defined(__GNUC__)
    return __builtin_ctzll(x);
#else
    int idx = 0;
    while ((x & 1ULL) == 0ULL) {
        x >>= 1;
        idx++;
    }
    return idx;
#endif
}

static int popcount64(U64 x) {
#if defined(__GNUC__)
    return __builtin_popcountll(x);
#else
    int c = 0;
    while (x) {
        x &= (x - 1ULL);
        c++;
    }
    return c;
#endif
}

static int bitset_empty(const U64 set[]) {
    int w;
    for (w = 0; w < WORDS; ++w) {
        if (set[w] != 0ULL) {
            return 0;
        }
    }
    return 1;
}

static int bitset_pop_first(U64 set[]) {
    int w;
    for (w = 0; w < WORDS; ++w) {
        if (set[w] != 0ULL) {
            U64 x = set[w];
            int b = bit_index64(x);
            set[w] = x & (x - 1ULL);
            return w * 64 + b;
        }
    }
    return -1;
}

// CORRIGIDO: Agora aloca estritamente até n_vertices sem vazar "vértices fantasmas"
static void set_all_candidates(U64 set[]) {
    int w;
    for (w = 0; w < WORDS; ++w) set[w] = 0ULL;
    for (int i = 0; i < n_vertices; ++i) {
        set[i >> 6] |= (1ULL << (i & 63));
    }
}

static void copy_bitset(U64 dst[], const U64 src[]) {
    memcpy(dst, src, sizeof(U64) * WORDS);
}

static void copy_best(unsigned short best[], int *best_size, const unsigned short current[], int size) {
    int i;
    *best_size = size;
    for (i = 0; i < size; ++i) {
        best[i] = current[i];
    }
}

static int cmp_degree_desc(const void *a, const void *b) {
    int va = *(const int *)a;
    int vb = *(const int *)b;

    if (old_degree[va] > old_degree[vb]) return -1;
    if (old_degree[va] < old_degree[vb]) return 1;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* ---------- input / output ---------- */

static void load_graph_and_reorder(const char *filename) {
    FILE *fp;
    char line[LINE_SIZE];
    int u, v, m;
    int i, new_i, new_j, old_i, old_j;
    int perm_local[MAXN];

    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ERROR: cannot open input file %s.\n", filename);
        exit(1);
    }

    n_vertices = 0;
    n_edges = 0;
    memset(old_adj, 0, sizeof(old_adj));
    memset(old_degree, 0, sizeof(old_degree));

    while (fgets(line, sizeof(line), fp) != NULL) {
        char type;
        if (sscanf(line, " %c", &type) != 1) continue;
        if (type == 'p') {
            if (sscanf(line, "p %*s %d %d", &n_vertices, &m) == 2) break;
        }
    }
    fclose(fp);

    if (n_vertices <= 0 || n_vertices > MAXN) {
        printf("ERROR: invalid DIMACS preamble or graph too large.\n");
        exit(1);
    }

    fp = fopen(filename, "r");
    while (fgets(line, sizeof(line), fp) != NULL) {
        char type;
        if (sscanf(line, " %c", &type) != 1) continue;

        if (type == 'e') {
            if (sscanf(line, "e %d %d", &u, &v) == 2) {
                u--; v--;
                if (u != v) {
                    if ((old_adj[u][v >> 6] & (1ULL << (v & 63))) == 0ULL) {
                        old_adj[u][v >> 6] |= (1ULL << (v & 63));
                        old_adj[v][u >> 6] |= (1ULL << (u & 63));
                        old_degree[u]++;
                        old_degree[v]++;
                        n_edges++;
                    }
                }
            }
        }
    }
    fclose(fp);

    for (i = 0; i < n_vertices; ++i) perm_local[i] = i;
    qsort(perm_local, n_vertices, sizeof(int), cmp_degree_desc);

    for (new_i = 0; new_i < n_vertices; ++new_i) {
        old_i = perm_local[new_i];
        perm[new_i] = old_i;
        original_label[new_i] = old_i + 1;
    }

    memset(adj, 0, sizeof(adj));
    for (new_i = 0; new_i < n_vertices; ++new_i) {
        old_i = perm[new_i];
        for (new_j = 0; new_j < n_vertices; ++new_j) {
            old_j = perm[new_j];
            if (old_adj[old_i][old_j >> 6] & (1ULL << (old_j & 63))) {
                adj[new_i][new_j >> 6] |= (1ULL << (new_j & 63));
            }
        }
    }
}

/* ---------- 1. Força Bruta ---------- */

static void bf_search(int v_idx, int size) {
    if (timeout_bf) return;

    calls_bf++;
    
    if ((calls_bf & 4095) == 0) {
        if ((double)(clock() - start_clock_bf) / CLOCKS_PER_SEC >= TIME_LIMIT_SECS) {
            timeout_bf = 1;
            return;
        }
    }

    if (size > best_size_bf) {
        copy_best(best_clique_bf, &best_size_bf, current_clique, size);
    }

    if (v_idx >= n_vertices) return;

    bf_search(v_idx + 1, size);
    if (timeout_bf) return;

    int can_add = 1;
    for (int i = 0; i < size; ++i) {
        int u = current_clique[i];
        if ((adj[u][v_idx >> 6] & (1ULL << (v_idx & 63))) == 0ULL) {
            can_add = 0;
            break;
        }
    }

    if (can_add) {
        current_clique[size] = (unsigned short)v_idx;
        bf_search(v_idx + 1, size + 1);
    }
}

static void solve_brute_force() {
    start_clock_bf = clock();
    best_size_bf = 0;
    calls_bf = 0;
    timeout_bf = 0;

    bf_search(0, 0);

    time_bf = (double)(clock() - start_clock_bf) / CLOCKS_PER_SEC;
    if (timeout_bf && time_bf > TIME_LIMIT_SECS) {
        time_bf = TIME_LIMIT_SECS;
    }
}

/* ---------- 2. Backtracking ---------- */

static void bt_search(U64 cand[], int size) {
    U64 remaining[WORDS], next[WORDS];
    int v, w;

    if (timeout_bt) return;

    calls_bt++;
    
    if ((calls_bt & 4095) == 0) {
        if ((double)(clock() - start_clock_bt) / CLOCKS_PER_SEC >= TIME_LIMIT_SECS) {
            timeout_bt = 1;
            return;
        }
    }

    if (size > best_size_bt) {
        copy_best(best_clique_bt, &best_size_bt, current_clique, size);
    }

    if (bitset_empty(cand)) return;
    copy_bitset(remaining, cand);

    while (!bitset_empty(remaining)) {
        if (timeout_bt) return;

        v = bitset_pop_first(remaining);
        if (v < 0) break;

        for (w = 0; w < WORDS; ++w) {
            next[w] = remaining[w] & adj[v][w];
        }

        current_clique[size] = (unsigned short)v;
        bt_search(next, size + 1);
    }
}

static void solve_backtracking() {
    U64 root[WORDS];
    start_clock_bt = clock();

    best_size_bt = 0;
    calls_bt = 0;
    timeout_bt = 0;

    set_all_candidates(root);
    bt_search(root, 0);

    time_bt = (double)(clock() - start_clock_bt) / CLOCKS_PER_SEC;
    if (timeout_bt && time_bt > TIME_LIMIT_SECS) {
        time_bt = TIME_LIMIT_SECS;
    }
}

/* ---------- 3. Branch and Bound ---------- */

// Abordagem Tomita Exata (Ordenação por Classe de Cor)
static void bb_search(U64 cand[], int size) {
    U64 remaining[WORDS], avail[WORDS];
    int colors = 0, w;
    int Q[MAXN];          // Armazena os vértices em cand
    int color_of[MAXN];   // Armazena a cor associada a cada vértice
    int q_size = 0;

    if (timeout_bb) return;

    calls_bb++;
    
    if ((calls_bb & 4095) == 0) {
        if ((double)(clock() - start_clock_bb) / CLOCKS_PER_SEC >= TIME_LIMIT_SECS) {
            timeout_bb = 1;
            return;
        }
    }

    if (size > best_size_bb) {
        copy_best(best_clique_bb, &best_size_bb, current_clique, size);
    }

    if (bitset_empty(cand)) return;

    copy_bitset(remaining, cand);

    // Passo 1: Coloração Gulosa para separar classes independentes (Heurística Tomita)
    while (!bitset_empty(remaining)) {
        colors++;
        copy_bitset(avail, remaining);
        U64 classset[WORDS] = {0};

        while (!bitset_empty(avail)) {
            int v = bitset_pop_first(avail);
            if (v < 0) break;

            classset[v >> 6] |= (1ULL << (v & 63));
            for (w = 0; w < WORDS; ++w) avail[w] &= ~adj[v][w];

            // Ordenando em classes e guardando o mapeamento
            Q[q_size] = v;
            color_of[q_size] = colors;
            q_size++;
        }

        for (w = 0; w < WORDS; ++w) remaining[w] &= ~classset[w];
    }

    // Passo 2: Expansão do Branch and Bound (Processando as cores de trás pra frente)
    copy_bitset(remaining, cand);

    for (int i = q_size - 1; i >= 0; i--) {
        if (timeout_bb) return;

        int v = Q[i];

        // PODA EXATA: Como iteramos da cor mais alta pra menor, se o tamanho atual
        // mais a cor deste nó não puder bater o best_size, todos os nós seguintes 
        // em Q (que têm cores iguais ou menores) também não baterão.
        if (size + color_of[i] <= best_size_bb) {
            prunes_bb++;
            return;
        }

        // Retira 'v' dos restantes para o nível atual (evita explorar permutações)
        remaining[v >> 6] &= ~(1ULL << (v & 63));

        U64 next[WORDS];
        for (w = 0; w < WORDS; ++w) {
            next[w] = remaining[w] & adj[v][w];
        }

        current_clique[size] = (unsigned short)v;
        bb_search(next, size + 1);
    }
}

static void solve_branch_and_bound() {
    U64 root[WORDS];
    start_clock_bb = clock();

    best_size_bb = 0;
    calls_bb = 0;
    prunes_bb = 0;
    timeout_bb = 0;

    set_all_candidates(root);
    bb_search(root, 0);

    time_bb = (double)(clock() - start_clock_bb) / CLOCKS_PER_SEC;
    if (timeout_bb && time_bb > TIME_LIMIT_SECS) {
        time_bb = TIME_LIMIT_SECS;
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s instancia.clq\n", argv[0]);
        return 1;
    }

    const char *csv_filename = "resultados_clique.csv";
    const char *instancia = argv[1];

    load_graph_and_reorder(instancia);

    const char *base = instancia;
    const char *s1 = strrchr(instancia, '/');
    const char *s2 = strrchr(instancia, '\\');
    if (s1 && s2) base = (s1 > s2) ? s1 + 1 : s2 + 1;
    else if (s1) base = s1 + 1;
    else if (s2) base = s2 + 1;

    printf("Processando: %s (V=%d, E=%d)\n", base, n_vertices, n_edges);

    solve_brute_force();
    solve_backtracking();
    solve_branch_and_bound();

    FILE *csv = fopen(csv_filename, "r");
    int write_header = (csv == NULL);
    if (csv) fclose(csv);

    csv = fopen(csv_filename, "a");
    if (!csv) {
        printf("ERRO: Nao foi possivel criar/abrir o arquivo CSV.\n");
        return 1;
    }

    if (write_header) {
        fprintf(csv, "Instancia,V,E,TempoLimite(s),MaxCliqueBF,NosBF,TempoBF(s),StatusBF,"
                     "MaxCliqueBT,NosBT,TempoBT(s),StatusBT,"
                     "MaxCliqueBB,NosBB,PodasBB,TempoBB(s),StatusBB\n");
    }

    fprintf(csv, "%s,%d,%d,%.0f,", base, n_vertices, n_edges, TIME_LIMIT_SECS);
    
    // Resultados Força Bruta
    fprintf(csv, "%d,%lld,%.6f,%s,", best_size_bf, calls_bf, time_bf, timeout_bf ? "TIMEOUT" : "OK");
    
    // Resultados Backtracking
    fprintf(csv, "%d,%lld,%.6f,%s,", best_size_bt, calls_bt, time_bt, timeout_bt ? "TIMEOUT" : "OK");

    // Resultados Branch and Bound
    fprintf(csv, "%d,%lld,%lld,%.6f,%s\n", best_size_bb, calls_bb, prunes_bb, time_bb, timeout_bb ? "TIMEOUT" : "OK");

    fclose(csv);
    printf("Resultados salvos em %s\n\n", csv_filename);

    return 0;
}
