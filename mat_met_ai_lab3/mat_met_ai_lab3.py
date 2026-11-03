# -*- coding: cp1251 -*-
import matplotlib.pyplot as plt
import numpy as np

# генерація даних для навчання

def generate_dataset(num_samples=1000, filename="dataset.csv"):
    np.random.seed(42)

    x1 = np.random.uniform(0, 1, num_samples)
    x2 = np.random.uniform(0, 1, num_samples)
    d = np.zeros(num_samples)

    for i in range(num_samples):
        # Умова належності до класу C1 (d=0) або C2 (d=1)
        if (x2[i] < x1[i] + 0.5) and (x2[i] > x1[i] - 0.5):
            d[i] = 0
        else:
            d[i] = 1

    data = np.column_stack((x1, x2, d))

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="x1,x2,d",
        comments="",
        fmt="%.4f,%.4f,%d"
    )

    print(f"[Генерація] Створено файл '{filename}' "
          f"з {num_samples} навчальними точками.")

    return x1, x2, d

# клас двошарового персептрона
class TwoLayerPerceptron:

    def __init__(self, weights=None, k=1.0, eta=1.0):

        self.k = k
        self.eta = eta

        if weights is not None:
            self.w10_1 = weights["w10_1"]
            self.w11_1 = weights["w11_1"]
            self.w12_1 = weights["w12_1"]

            self.w20_1 = weights["w20_1"]
            self.w21_1 = weights["w21_1"]
            self.w22_1 = weights["w22_1"]

            self.w10_2 = weights["w10_2"]
            self.w11_2 = weights["w11_2"]
            self.w12_2 = weights["w12_2"]

        else:
            
            self.w10_1 = 0.1
            self.w11_1 = -0.3
            self.w12_1 = 0.4

            self.w20_1 = -0.7
            self.w21_1 = -0.1
            self.w22_1 = 0.01

            self.w10_2 = 0.4
            self.w11_2 = -0.2
            self.w12_2 = 0.1

        # Фіксовані зміщення (bias)
        self.b1_1 = 1.0
        self.b2_1 = 1.0
        self.b1_2 = 1.0

    # Функція активації
    def activation(self, u):
        return 1.0 / (1.0 + np.exp(-self.k * u))

    # Похідна функції активації
    def activation_derivative(self, y):
        return y * self.k * (1.0 - y)

    # Прямий прохід
    def forward(self, x1, x2):

        # Прихований шар
        u1_1 = (
            self.w10_1 * self.b1_1
            + self.w11_1 * x1
            + self.w12_1 * x2
        )

        u2_1 = (
            self.w20_1 * self.b2_1
            + self.w21_1 * x1
            + self.w22_1 * x2
        )

        y1_1 = self.activation(u1_1)
        y2_1 = self.activation(u2_1)

        # Вихідний шар
        u1_2 = (
            self.w10_2 * self.b1_2
            + self.w11_2 * y1_1
            + self.w12_2 * y2_1
        )

        y = self.activation(u1_2)

        return y1_1, y2_1, y, u1_1, u2_1, u1_2

    # Навчання протягом однієї епохи
    def train_epoch(self, X1, X2, D):

        total_error = 0.0
        K = len(D)

        for i in range(K):

            x1 = X1[i]
            x2 = X2[i]
            d = D[i]

            # Прямий прохід
            y1_1, y2_1, y, u1_1, u2_1, u1_2 = self.forward(x1, x2)

            # Похибка
            eps = d - y
            total_error += 0.5 * (eps ** 2)

            # Похідні функцій активації
            f_prime_u1_2 = self.activation_derivative(y)
            f_prime_u1_1 = self.activation_derivative(y1_1)
            f_prime_u2_1 = self.activation_derivative(y2_1)

            # Зворотне поширення похибки
            eta_eps_f2 = self.eta * eps * f_prime_u1_2

            # Зберігаємо старі ваги вихідного шару
            old_w11_2 = self.w11_2
            old_w12_2 = self.w12_2

            # Обчислюємо корекції ваг прихованого шару
            common1 = (
                eta_eps_f2 * old_w11_2 * f_prime_u1_1
            )

            common2 = (
                eta_eps_f2 * old_w12_2 * f_prime_u2_1
            )

            # Ваги вихідного шару
            self.w10_2 += eta_eps_f2 * self.b1_2
            self.w11_2 += eta_eps_f2 * y1_1
            self.w12_2 += eta_eps_f2 * y2_1

            # Ваги прихованого шару (нейрон 1)
            self.w10_1 += common1 * self.b1_1
            self.w11_1 += common1 * x1
            self.w12_1 += common1 * x2

            # Ваги прихованого шару (нейрон 2)
            self.w20_1 += common2 * self.b2_1
            self.w21_1 += common2 * x1
            self.w22_1 += common2 * x2

        return total_error / K

    # Тестування на чотирьох точках XOR
    def test_xor(self):

        test_points = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0)
        ]

        e_star = 0.0

        for x1, x2, d in test_points:

            _, _, y, _, _, _ = self.forward(x1, x2)

            e_star += 0.5 * ((d - y) ** 2)

        return e_star

    # Збереження поточних ваг
    def get_weights(self):

        return {
            "w10_1": self.w10_1,
            "w11_1": self.w11_1,
            "w12_1": self.w12_1,
            "w20_1": self.w20_1,
            "w21_1": self.w21_1,
            "w22_1": self.w22_1,
            "w10_2": self.w10_2,
            "w11_2": self.w11_2,
            "w12_2": self.w12_2
        }


# виконання досліджень

if __name__ == "__main__":

    # Генерація навчального датасету
    X1, X2, D = generate_dataset(num_samples=1000)

    epochs = 200
    threshold = 0.01

    # дослідження 1
    print("\nДослідження 1: Рекомендовані параметри")

    nn_rec = TwoLayerPerceptron(k=1.0, eta=1.0)

    epoch_errors = []
    test_errors = []

    best_error = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, epochs + 1):

        train_err = nn_rec.train_epoch(X1, X2, D)
        test_err = nn_rec.test_xor()

        epoch_errors.append(train_err)
        test_errors.append(test_err)

        # Збереження найкращих ваг
        if test_err < best_error:

            best_error = test_err
            best_epoch = epoch
            best_weights = nn_rec.get_weights()

        if epoch % 10 == 0 or epoch == 1:

            print(
                f"Епоха {epoch:3d} | "
                f"Середня похибка навчання: {train_err:.6f} | "
                f"Тестова похибка E*: {test_err:.6f}"
            )

    optimal_epoch = best_epoch

    print("\nНайкраща кількість епох:", optimal_epoch)
    print(f"Мінімальна тестова похибка E*: {best_error:.6f}")

    print("\nНайкращі вагові коефіцієнти:")

    for name, value in best_weights.items():
        print(f"{name} = {value:.6f}")

    recommended_epoch = next(
        (i + 1 for i, e in enumerate(test_errors)
         if e < threshold),
        None
    )

    if recommended_epoch is not None:
        print(
            f"Кількість епох для досягнення E* < {threshold}: "
            f"{recommended_epoch}"
        )
    else:
        print(
            f"За {epochs} епох похибка не стала меншою за {threshold}"
        )

    # дослідження 2 вплив початкових ваг

    print("\nДослідження 2: Випадкові початкові ваги")

    np.random.seed(0)

    random_weights = {
        "w10_1": np.random.uniform(-1, 1),
        "w11_1": np.random.uniform(-1, 1),
        "w12_1": np.random.uniform(-1, 1),

        "w20_1": np.random.uniform(-1, 1),
        "w21_1": np.random.uniform(-1, 1),
        "w22_1": np.random.uniform(-1, 1),

        "w10_2": np.random.uniform(-1, 1),
        "w11_2": np.random.uniform(-1, 1),
        "w12_2": np.random.uniform(-1, 1)
    }

    nn_rand = TwoLayerPerceptron(
        weights=random_weights,
        k=1.0,
        eta=1.0
    )

    rand_train_errors = []
    rand_test_errors = []

    for epoch in range(1, epochs + 1):

        train_err = nn_rand.train_epoch(X1, X2, D)
        test_err = nn_rand.test_xor()

        rand_train_errors.append(train_err)
        rand_test_errors.append(test_err)

    random_epoch = next(
        (i + 1 for i, e in enumerate(rand_test_errors)
         if e < threshold),
        None
    )

    print(f"Випадкові ваги — похибка на останній епосі: "
          f"{rand_test_errors[-1]:.6f}")

    print(f"Мінімальна тестова похибка: "
          f"{min(rand_test_errors):.6f}")

    if random_epoch is not None:
        print(
            f"Кількість епох для досягнення E* < {threshold}: "
            f"{random_epoch}"
        )
    else:
        print(
            f"За {epochs} епох випадкова мережа "
            f"не досягла порога E* < {threshold}"
        )

    print("\nПорівняння початкових ваг:")

    print(
        "Рекомендовані ваги:",
        recommended_epoch if recommended_epoch is not None
        else "Поріг не досягнуто"
    )

    print(
        "Випадкові ваги:",
        random_epoch if random_epoch is not None
        else "Поріг не досягнуто"
    )

    #дослідження 3вплив коєфіцієнта швидкості навчання eta
    print("\n Дослідження 3: Вплив коефіцієнта eta")

    etas = [0.1, 0.3, 0.5, 0.7, 1.0]

    plt.figure(figsize=(10, 6))

    for et in etas:

        nn_eta = TwoLayerPerceptron(k=1.0, eta=et)

        errs = []

        for _ in range(30):
            err = nn_eta.train_epoch(X1, X2, D)
            errs.append(err)

        plt.plot(
            range(1, 31),
            errs,
            label=f"eta = {et}"
        )

        print(
            f"eta = {et} | "
            f"Похибка після 30 епох: {errs[-1]:.6f}"
        )

    plt.title("Вплив швидкості навчання eta")
    plt.xlabel("Кількість епох")
    plt.ylabel("Середня похибка навчання")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # дослідження 4 вплив параметру крутизни k

    print("\nДослідження 4: Вплив параметра k")

    ks = [0.5, 1.0, 2.0, 5.0]

    plt.figure(figsize=(10, 6))

    for kval in ks:

        nn_k = TwoLayerPerceptron(k=kval, eta=1.0)

        errs = []

        for _ in range(30):
            err = nn_k.train_epoch(X1, X2, D)
            errs.append(err)

        plt.plot(
            range(1, 31),
            errs,
            label=f"k = {kval}"
        )

        print(
            f"k = {kval} | "
            f"Похибка після 30 епох: {errs[-1]:.6f}"
        )

    plt.title("Вплив параметра крутизни k")
    plt.xlabel("Кількість епох")
    plt.ylabel("Середня похибка навчання")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Дослідження 5 Залежність E* від кількості епох

    print("\n Дослідження 5: Залежність E* від кількості епох")

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, epochs + 1),
        test_errors,
        "b-o",
        markersize=3,
        label="Тестова похибка E*"
    )

    plt.axvline(
        x=optimal_epoch,
        color="red",
        linestyle="--",
        label=f"Оптимальна епоха: {optimal_epoch}"
    )

    plt.scatter(
        optimal_epoch,
        best_error,
        color="red",
        zorder=5
    )

    plt.title("Залежність тестової похибки E* від кількості епох")
    plt.xlabel("Кількість епох навчання")
    plt.ylabel("Тестова похибка E*")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # додатковий графік орівняння впливу початкових ваг

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, epochs + 1),
        test_errors,
        label="Рекомендовані ваги"
    )

    plt.plot(
        range(1, epochs + 1),
        rand_test_errors,
        label="Випадкові ваги"
    )

    plt.title("Порівняння впливу початкових ваг")
    plt.xlabel("Кількість епох")
    plt.ylabel("Тестова похибка E*")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("ВИКОНАННЯ ЛАБОРАТОРНОЇ РОБОТИ ЗАВЕРШЕНО")
  
