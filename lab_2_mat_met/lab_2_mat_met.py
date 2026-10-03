import math
import matplotlib.pyplot as plt

#константи і параметри моделювання
dt = 0.01          # крок інтегрування [с]
T = 50.0           # тривалість моделювання [с]
d_mu_tilde = 0.1   # крок дефазифікації 

# Діапазони змінних
phi_min, phi_max = -math.pi, math.pi               # [-180, 180] deg
omega_min, omega_max = -math.pi / 9, math.pi / 9   # [-20, 20] deg/s
mu_min, mu_max = -math.pi / 36, math.pi / 36       # [-5, 5] deg/s^2

# База правил (Таблиця 2.1)
RULE_BASE = [
    (1, 1, 6), (1, 2, 6), (1, 3, 6), (1, 4, 6), (1, 5, 6), (1, 6, 6),
    (2, 1, 6), (2, 2, 6), (2, 3, 6), (2, 4, 5), (2, 5, 5), (2, 6, 5),
    (3, 1, 6), (3, 2, 6), (3, 3, 6), (3, 4, 5), (3, 5, 5), (3, 6, 5),
    (4, 1, 2), (4, 2, 2), (4, 3, 2), (4, 4, 1), (4, 5, 1), (4, 6, 1),
    (5, 1, 2), (5, 2, 2), (5, 3, 2), (5, 4, 1), (5, 5, 1), (5, 6, 1),
    (6, 1, 1), (6, 2, 1), (6, 3, 1), (6, 4, 1), (6, 5, 1), (6, 6, 1)
]


#функції нечіткої логіки
def F_T(x, a, b, c, d):
    """ Універсальна трапецієподібна/трикутна/Z/S функція належності (формула 2.1) """
    if x <= a:
        return 0.0
    elif a <= x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b <= x <= c:
        return 1.0
    elif c <= x <= d:
        return (d - x) / (d - c) if d != c else 1.0
    else:
        return 0.0


def get_membership(term_idx, x):
    """ Отримання значення функції належності за номером терму (1..6) """
    if term_idx == 1:   # HB
        return F_T(x, -1000, -200, -100, -50)
    elif term_idx == 2: # HC
        return F_T(x, -100, -50, -50, -10)
    elif term_idx == 3: # HM
        return F_T(x, -50, -10, 0, 0)
    elif term_idx == 4: # PM
        return F_T(x, 0, 0, 10, 50)
    elif term_idx == 5: # PS
        return F_T(x, 10, 50, 50, 100)
    elif term_idx == 6: # PB
        return F_T(x, 50, 100, 200, 1000)
    return 0.0


def fuzzy_inference(phi, omega):
    """ Нечітке виведення (Рис. 3.1, 3.2) """
    phi_tilde = (200.0 / (phi_max - phi_min)) * (phi - phi_min) - 100.0
    omega_tilde = (200.0 / (omega_max - omega_min)) * (omega - omega_min) - 100.0

    # Попередньо вираховуєтся активність правил для прискорення
    rule_gammas = []
    for n1, n2, m in RULE_BASE:
        alpha = get_membership(n1, phi_tilde)
        beta = get_membership(n2, omega_tilde)
        gamma = min(alpha, beta)
        if gamma > 0:
            rule_gammas.append((gamma, m))

    s1, s2 = 0.0, 0.0
    mu_tilde = -100.0

    while mu_tilde <= 100.0:
        chi_star = 0.0
        for gamma, m in rule_gammas:
            delta = get_membership(m, mu_tilde)
            chi = min(gamma, delta)
            if chi > chi_star:
                chi_star = chi

        s1 += mu_tilde * chi_star * d_mu_tilde
        s2 += chi_star * d_mu_tilde
        mu_tilde += d_mu_tilde

    mu_tilde_star = (s1 / s2) if abs(s2) > 1e-9 else 0.0
    return ((mu_tilde_star + 100.0) / 200.0) * (mu_max - mu_min) + mu_min


# основний цикл моделювання
def run_simulation(phi0_deg, omega0_deg, use_classical=False):
    phi = math.radians(phi0_deg)
    omega = math.radians(omega0_deg)
    t = 0.0

    s1_root, s2_root = -1.0, -0.7
    k1 = s1_root + s2_root
    k2 = -s1_root * s2_root

    history_t, history_phi, history_omega, history_mu = [], [], [], []

    while t < (T - 0.5 * dt):
        history_t.append(t)
        history_phi.append(math.degrees(phi))
        history_omega.append(math.degrees(omega))

        if use_classical:
            mu = k1 * omega + k2 * phi
            mu = max(mu_min, min(mu_max, mu))
        else:
            mu = fuzzy_inference(phi, omega)

        history_mu.append(math.degrees(mu))

        phi += omega * dt
        omega += mu * dt
        t += dt

    return history_t, history_phi, history_omega, history_mu


# блок виконання
if __name__ == "__main__":
    VAR_PHI0 = 25.0   # градуси
    VAR_OMEGA0 = 2.0  # град/с

    print(f"Запуск симуляції: phi0 = {VAR_PHI0} deg, omega0 = {VAR_OMEGA0} deg/s...")

    t_fuz, phi_fuz, omega_fuz, mu_fuz = run_simulation(VAR_PHI0, VAR_OMEGA0, use_classical=False)

    with open("simulation_results.txt", "w", encoding="utf-8") as f:
        f.write("Time(s)\tPhi(deg)\tOmega(deg/s)\tMu(deg/s^2)\n")
        for i in range(len(t_fuz)):
            f.write(f"{t_fuz[i]:.2f}\t{phi_fuz[i]:.4f}\t{omega_fuz[i]:.4f}\t{mu_fuz[i]:.4f}\n")

    t_cls, phi_cls, omega_cls, mu_cls = run_simulation(VAR_PHI0, VAR_OMEGA0, use_classical=True)

    print("Обчислення завершено. Відображення графіків...")

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(t_fuz, phi_fuz, 'r-', label='Нечіткий регулятор')
    plt.plot(t_cls, phi_cls, 'b--', label='Класичний регулятор')
    plt.ylabel('Кут φ, [град]')
    plt.title('Порівняння нечіткого та класичного алгоритмів управління ШСЗ')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_fuz, omega_fuz, 'r-', label='Нечіткий регулятор')
    plt.plot(t_cls, omega_cls, 'b--', label='Класичний регулятор')
    plt.ylabel('Швидкість ω, [град/с]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t_fuz, mu_fuz, 'r-', label='Нечіткий регулятор')
    plt.plot(t_cls, mu_cls, 'b--', label='Класичний регулятор')
    plt.xlabel('Час t, [с]')
    plt.ylabel('Прискорення μ, [град/с²]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()