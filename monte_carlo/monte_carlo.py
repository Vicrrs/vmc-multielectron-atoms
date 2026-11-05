"""
O Algoritmo de Monte Carlo é uma técnica estatística que usa numeros aleatorios para resolver problemas 
determinísticos. Em temos simples, se precisarmos repetir um experimento aleatório milhares de vezes, 
a média dos resultados vai convergir para a resposta correta.

O exemplo mais clássico e didático para aprender é estimar o valor de Pi (π) desenhando um círculo dentro 
de um quadrado.

1. Imagine um quadrado de lado 2. A área dele é 2x2=4.
2. Dentro dele, desenhamos um círculo de raio 1. A área dele é π × 1² = π.
3. Se jogarmos dardos aleatoriamente nesse quadrado, a chance de acertar dentro do círculo é a proporção entre as duas areas:

(area do círculo)/(area do quadrado) = π/4

4. Por tanto, isolando Pi:

π = 4 * (Dados dentro do círculo)/(Total de dardos sorteados)
"""

import random

def estimar_pi(num_pontos):
    pontos_no_circulo = 0

    for _ in range(num_pontos):
        # Sorteia coordenadas X e Y entre -1 e 1
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # Distância do ponto até a origem (0,0) usando Pitágoras: x² + y²
        # Se for menor ou igual a 1, o ponto está dentro do círculo
        if x**2 + y**2 <= 1:
            pontos_no_circulo += 1

    # Aplica a fórmula matemática de Monte Carlo
    pi_estimado = 4 * (pontos_no_circulo / num_pontos)
    return pi_estimado

# Testando o algoritmo com diferentes quantidades de pontos
print(f"Com 1.000 pontos: {estimar_pi(1000)}")
print(f"Com 100.000 pontos: {estimar_pi(100000)}")
print(f"Com 10.000.000 pontos: {estimar_pi(10000000)}")
