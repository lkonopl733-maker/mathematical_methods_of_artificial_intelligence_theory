import math
import matplotlib.pyplot as plt
import numpy as np
a, b = -2.0, 4.0  # Межі для x
c, d = -4.0, 0.0  # Межі для y
q = 1  # Точність 0.1

# Розрахунок довжини хромосом (Lx, Ly) та фактичних кроків квантування (hx, hy)
Lx = math.ceil(math.log2((b - a) * (10**q) + 1))  # Lx = 6
Ly = math.ceil(math.log2((d - c) * (10**q) + 1))  # Ly = 6

hx = (b - a) / (2**Lx - 1)  # hx = 0.0952 <= 0.1
hy = (d - c) / (2**Ly - 1)  # hy = 0.0635 <= 0.1

# Параметри генетичного алгоритму 
POP_SIZE = 50  # Розмір популяції 
NUM_ITERATIONS = 500  # Кількість ітерацій 
CROSSOVER_PROB = 0.9  # Ймовірність схрещування
MUTATION_PROB = 0.01  # Ймовірність мутації



def fitness_func(x, y):
    """Цільова функція: f(x, y) = |x| + y"""
    return abs(x) + y


def decode(chrom_x, chrom_y):
    """Декодування двійкових хромосом у дійсні координати (x, y)"""
    x_int = int("".join(map(str, chrom_x)), 2)
    y_int = int("".join(map(str, chrom_y)), 2)
    x = a + x_int * hx
    y = c + y_int * hy
    return x, y


# Ініціалізація початкової популяції 
np.random.seed(42)  # Фіксація генератора для відтворюваності
pop_x = np.random.randint(0, 2, size=(POP_SIZE, Lx))
pop_y = np.random.randint(0, 2, size=(POP_SIZE, Ly))

best_fitness_history = []
best_overall_fit = -float("inf")
best_overall_x = None
best_overall_y = None

# основ. цикл генетичного алгоритму
for iter_idx in range(NUM_ITERATIONS):
    # Декод. та обчис. пристосованості 
    fitness_values = []

    for i in range(POP_SIZE):
        x, y = decode(pop_x[i], pop_y[i])
        fit = fitness_func(x, y)
        fitness_values.append(fit)

        # Оновлення абсолютного max за весь час
        if fit > best_overall_fit:
            best_overall_fit = fit
            best_overall_x = x
            best_overall_y = y

    # Запис найкращого значення функції на поточній ітерації
    best_fitness_history.append(np.max(fitness_values))

    # Селекція рулеткою
    fit_arr = np.array(fitness_values)
    min_fit = np.min(fit_arr)
    if min_fit <= 0:
        fit_shifted = fit_arr + 2 * abs(min_fit) + 1e-6
    else:
        fit_shifted = fit_arr.copy()

    probs = fit_shifted / np.sum(fit_shifted)
    selected_indices = np.random.choice(POP_SIZE, size=POP_SIZE, p=probs)

    selected_x = pop_x[selected_indices].copy()
    selected_y = pop_y[selected_indices].copy()

    # Створення нової популяції схрещ. і мутація
    next_x, next_y = [], []

    for i in range(0, POP_SIZE, 2):
        p1_x, p1_y = selected_x[i].copy(), selected_y[i].copy()
        p2_x, p2_y = selected_x[i + 1].copy(), selected_y[i + 1].copy()

        # Оператор схрещування (одноточковий кросовер)
        if np.random.rand() < CROSSOVER_PROB:
            kx = np.random.randint(1, Lx)
            ky = np.random.randint(1, Ly)

            c1_x = np.concatenate([p1_x[:kx], p2_x[kx:]])
            c2_x = np.concatenate([p2_x[:kx], p1_x[kx:]])
            c1_y = np.concatenate([p1_y[:ky], p2_y[ky:]])
            c2_y = np.concatenate([p2_y[:ky], p1_y[ky:]])
        else:
            c1_x, c2_x = p1_x, p2_x
            c1_y, c2_y = p1_y, p2_y

        # Оператор мутації (інверсія біта)
        for child_x in (c1_x, c2_x):
            for bit in range(Lx):
                if np.random.rand() < MUTATION_PROB:
                    child_x[bit] = 1 - child_x[bit]

        for child_y in (c1_y, c2_y):
            for bit in range(Ly):
                if np.random.rand() < MUTATION_PROB:
                    child_y[bit] = 1 - child_y[bit]

        next_x.extend([c1_x, c2_x])
        next_y.extend([c1_y, c2_y])

    pop_x = np.array(next_x)
    pop_y = np.array(next_y)



print("Результати роботи генетичного алгоритму")
print(f"Максимальне значення функції max f(x, y): {best_overall_fit:.4f}")
print(f"Координати найкращої точки: x = {best_overall_x:.4f}, y = {best_overall_y:.4f}")


#Побудова графіка еволюції найкращого пристосування
plt.figure(figsize=(10, 5))
plt.plot(best_fitness_history, color="blue", linewidth=1.5, label="Best Fitness")
plt.title("Динаміка найкращого значення функції f(x, y) на кожній ітерації")
plt.xlabel("Ітерація (Покоління)")
plt.ylabel("max f(x, y)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()
