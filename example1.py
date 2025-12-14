import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 1. Заданные параметры и матрицы
n = 2
m = 2
l = 2

# Матрицы из примера
A = np.array([[1, 3],
              [2, 4]])
b = np.array([[5],
              [6]])
F = np.eye(2)         # F = I_2
D = np.zeros((l, m))  # D = 0 (l x m)
E_A = np.array([[0, 1],
                [0, 0]])
E_b = np.array([[0],
                [1]])

# Параметры симуляции
rho_values = np.arange(0, 2.1, 0.1)  # от 0 до 2 с шагом 0.1

# Списки для хранения результатов
gamma_results = []
x1_results = []
x2_results = []
valid_rhos = []  # Для хранения rho, где решение было найдено

print(f"Запуск расчета для {len(rho_values)} точек...")

for rho in rho_values:
    # Обновляем M для текущего шага
    M = rho * np.eye(l)

    # 2. Переменные оптимизации
    # x - вектор-столбец (n, 1)
    x = cp.Variable((n, 1))
    # Lambda - скаляр > 0
    lam = cp.Variable(nonneg=True)
    # nu - скаляр > 0 (nu = gamma^2)
    nu = cp.Variable(nonneg=True)

    # --- Подготовка блоков для матрицы ЛМН ---
    
    # Вспомогательные выражения
    Ax_b = A @ x - b          # Размер (n, 1)
    EAx_Eb = E_A @ x - E_b    # Размер (l, 1)

    # -- Строка 1 --
    # [-I, lam*F, Ax-b, 0]
    row1 = [
        -np.eye(n),
        F * lam,
        Ax_b,
        np.zeros((n, l))  # O_{n x l}
    ]

    # -- Строка 2 --
    # [lam*F.T, -lam*I, 0, lam*D.T*M.T]
    # Примечание: так как D=0, член lam*D.T*M.T будет нулевым, но оставим формулу для общности
    row2 = [
        lam * F.T,
        -lam * np.eye(m),
        np.zeros((m, 1)),   # O_{m \times 1}
        lam * D.T @ M.T     # Размер (m, l)
    ]

    # -- Строка 3 --
    # [(Ax-b).T, 0, -nu, (EAx-Eb).T * M.T]
    row3 = [
        Ax_b.T,
        np.zeros((1, m)),   # O_{1 \times m}
        -cp.reshape(nu, (1, 1)), # Приводим скаляр к матрице (1,1)
        EAx_Eb.T @ M.T      # Размер (1, l)
    ]

    # -- Строка 4 --
    # [0, M*D*lam, M(EAx-Eb), -lam*I]
    row4 = [
        np.zeros((l, n)),   # O_{l \times n}
        M @ D * lam,        # Размер (l, m)
        M @ EAx_Eb,         # Размер (l, 1)
        -lam * np.eye(l)
    ]

    # Сборка матрицы
    LMI = cp.bmat([
        row1, 
        row2, 
        row3, 
        row4
    ])

    # 3. Ограничения и Целевая функция
    # LMI должна быть отрицательно полуопределенной (<< 0)
    # Добавляем lam >= eps и nu >= eps для строгой положительности
    constraints = [
        LMI << 0,
        lam >= 1e-6,
        nu >= 1e-6
    ]

    objective = cp.Minimize(nu)
    prob = cp.Problem(objective, constraints)

    try:
        # Решаем задачу
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status == 'optimal':
            # Сохраняем результаты
            current_gamma = np.sqrt(nu.value)
            gamma_results.append(current_gamma)
            
            # x.value имеет размер (2,1), берем элементы
            x_val = x.value.flatten()
            x1_results.append(x_val[0])
            x2_results.append(x_val[1])
            valid_rhos.append(rho)
        else:
            print(f"rho={rho:.1f}: Решение не найдено (статус {prob.status})")

    except Exception as e:
        print(f"rho={rho:.1f}: Ошибка солвера {e}")

# 4. Визуализация
if len(valid_rhos) > 0:
    plt.figure(figsize=(12, 5))

    # График 1: Гамма от Ро
    plt.subplot(1, 2, 1)
    plt.plot(valid_rhos, gamma_results, 'b-o', linewidth=2)
    plt.title(r'Зависимость $\gamma$ от $\rho$')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\gamma = \sqrt{\nu}$')
    plt.grid(True)

    # График 2: Фазовая траектория x1 vs x2
    plt.subplot(1, 2, 2)
    plt.plot(x1_results, x2_results, 'r-o', linewidth=2, markersize=4, label='Траектория')
    
    # Отметим начало и конец
    plt.scatter(x1_results[0], x2_results[0], color='green', s=100, zorder=5, label=f'Start (rho={valid_rhos[0]:.1f})')
    plt.scatter(x1_results[-1], x2_results[-1], color='black', s=100, zorder=5, label=f'End (rho={valid_rhos[-1]:.1f})')
    
    plt.title(r'Траектория вектора $x$ на плоскости $(x_1, x_2)$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Важно для корректного отображения геометрии

    plt.tight_layout()
    
    # Сохраняем результат
    filename = 'lmi_solution.png'
    plt.savefig(filename)
    print(f"\nГрафики построены и сохранены в файл: {filename}")
    
    # Вывод числовых значений для проверки
    print(f"Min gamma (rho=0): {gamma_results[0]:.4f}")
    print(f"Max gamma (rho=2): {gamma_results[-1]:.4f}")
else:
    print("Не удалось получить валидные решения.")