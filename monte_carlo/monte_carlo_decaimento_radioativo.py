"""
Na física nuclear, o decaimento radioativo é um processo puramente probabilístico.
Não sabemos exatamente quando um átomo específico vai decair, mas sabemos a probabilidade 
de ele decair em um determinado intervalo de tempo (baseado na sua meia vida).


1. Começamos com um grupo de N átomo instáveis

2. A cada "segundo" (passo de tempo), cada átomo joga um dado (gera um numero aleatório  entre 0 e 1).

3. Se o número for menor que a probabilidade de decaimento, o átomo decai e transforma em um elemento estável.

4. Repetimos isso ao longo do empo para ver a curva de decaimento exponencial surgir naturalmente.
"""

import random

def simular_decaimento(atomos_iniciais, chance_decaimento, passos_tempo):
    atomos_restantes = atomos_iniciais

    print(f"Tempo: 0 s | Átomos restantes: {atomos_restantes}")

    for segundo in range(1, passos_tempo + 1):
        atomos_que_decairam = 0

        # Monte Carlo: testa a chance de cada átomo decair de forma independente
        for _ in range(atomos_restantes):
            if random.random() < chance_decaimento:
                atomos_que_decairam += 1

        # Atualiza a quantidade para o próximo segundo
        atomos_restantes -= atomos_que_decairam

        # Mostra o progresso a cada passo relevante
        if segundo % 5 == 0 or atomos_restantes == 0:
            print(f"Tempo: {segundo} s | Átomos restantes: {atomos_restantes}")

        if atomos_restantes == 0:
            print("Todos os átomos decaíram.")
            break

# Configuração da Simulação
# Começamos com 10.000 átomos, onde cada um tem 10% (0.1) de chance de decair por segundo
simular_decaimento(atomos_iniciais=10000, chance_decaimento=0.1, passos_tempo=50)

