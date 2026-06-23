

implementação em linguagem C de algoritmos exatos para a resolução do Problema da Clique Máxima (NP-Completo). O código avalia e compara o desempenho de três abordagens:
1. **Força Bruta (Busca Exaustiva)**
2. **Backtracking Lógico**
3. **Branch and Bound (otimizado com Heurística de Coloração Gulosa MCQ)**

Os testes são focados em instâncias de grafos clássicas da literatura (DIMACS Benchmark).


Linux

gcc clique_maximo.c -o clique_maximo -O3
gcc conversor.c -o conversor -O3

Windows

gcc clique_maximo.c -o clique_maximo.exe -O3
gcc conversor.c -o conversor.exe -O3

Como Usar
1. Preparação dos Dados (Conversão Binário para ASCII)
Algumas instâncias DIMACS (como .clq.b) são compactadas em formato binário. O executável que fiz lê apenas formato texto (ASCII). Para converter:

./bin2asc nome_da_instancia.b

o resultado da conversão será um arquivo .clq (arquivo texto) que serve de entrada para o clique_maximo.c