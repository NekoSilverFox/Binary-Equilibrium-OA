# ------*------ coding: utf-8 ------*------
# @Time    : 2022/11/9 15:41
# @Author  : Мэн Цзянин (冰糖雪狸 | NekoSilverfox)
# @Project : Binary-Equilibrium-OA
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from enum import Enum
from math import atan, sqrt, tanh, erf, e, pi
from KP import *


def blockPrint():
    """
    Disable print
    :return: None
    """
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """
    Restore print
    :return: None
    """
    sys.stdout = sys.__stdout__


class TransferFuncion(Enum):
    """значения для передаточных функций V-Shaped и S-Shapes"""
    V1 = 1
    V2 = 2
    V3 = 3
    V4 = 4
    S1 = 5
    S2 = 6
    S3 = 7
    S4 = 8


def transfer_function(transfer_function_type: TransferFuncion, a: float) -> float:
    """8 передаточных функций (4 V-Shapes и 4 S-Shapes), которые отображают значения на интервал [0, 1]

    Args:
        transfer_function_type (Transfer Function): Тип передаточной функции
        a (float): Произвольные реальные значения

    Returns:
        float: Значения, отображаемые на интервал [0, 1] передаточной функцией
    """
    if transfer_function_type == TransferFuncion.V1:
        return abs((2 / pi) * atan((pi / 2) * a))

    elif transfer_function_type == TransferFuncion.V2:
        return abs(tanh(a))

    elif transfer_function_type == TransferFuncion.V3:
        return abs(a / (sqrt(1 + a ** 2)))

    elif transfer_function_type == TransferFuncion.V4:
        return abs(erf((sqrt(pi) / 2) * a))

    elif transfer_function_type == TransferFuncion.S1:
        return 1 / (1 + e ** (-a))

    elif transfer_function_type == TransferFuncion.S2:
        return 1 / (1 + e ** (-2 * a))

    elif transfer_function_type == TransferFuncion.S3:
        return 1 / (1 + (e ** (-a / 2)))

    elif transfer_function_type == TransferFuncion.S4:
        return 1 / (1 + (e ** (-a / 3)))

    else:
        print('[ERROR] Unknown transfer function type, Please use V1~V4 or S1~S4')
        exit


def initialization(Particles_no: int, dim: int) -> np.ndarray:
    """Сгенерируйте случайный бинарный массив Particles_no строк и dim столбцов C

    Args:
        Particles_no (int): _description_
        dim (int): _description_

    Returns:
        np.ndarray: _description_
    """
    C = np.zeros(shape=(Particles_no, dim))

    for i in range(Particles_no):
        for j in range(dim):
            if np.random.rand() > 0.5:
                C[i, j] = 0
            else:
                C[i, j] = 1

    return C


def arr2bin(arr: np.ndarray, tf: TransferFuncion) -> np.ndarray:
    for i in range(arr.shape[0]):
        if transfer_function(tf, arr[i]) >= np.random.rand():
            arr[i] = 0
        else:
            arr[i] = 1
    return arr


def get_price_table(n: int, min_price: int, max_price: int) -> np.ndarray:
    """Сгенерируйте таблицу случайных значений элементов (n строк, 1 столбец),
    т.е. соответствующих n элементам. Диапазон значений: [min_price, max_price].

    Args:
        n (int): Количество предметов
        min_price (int): Минимальное значение (больше 0)
        max_price (int): Максимальное значение (больше минимального)

    Returns:
        np.ndarray: Одномерные массивы
    """
    if n < 0:
        print('[ERROR] n CAN NOT small than 1')
        exit

    if (min_price < 1) or (max_price <= min_price):
        print('[ERROR] min_price CAN NOT small than 1 and max_price CAN NOT small than min_price')
        exit

    return np.random.randint(low=min_price + 1, high=max_price + 1, size=n)


def get_weight_table(n: int, min_weight: int, max_weight: int) -> np.ndarray:
    """Сгенерируйте случайную таблицу весов элементов (n строк, 1 столбец),
    т.е. соответствующих n элементам. Диапазон весов: [min_weight, max_weight].

    Args:
        n (int): Количество предметов
        min_weight (int): Минимальный вес (больше 0)
        max_weight (int): Максимальный вес (больше минимального)

    Returns:
        np.ndarray: Одномерные массивы
    """
    if n < 0:
        print('[ERROR] n CAN NOT small than 1')
        exit

    if (min_weight < 1) or (max_weight <= min_weight):
        print('[ERROR] min_weight CAN NOT small than 1 and max_weight CAN NOT small than min_weight')
        exit

    return np.random.randint(low=min_weight + 1, high=max_weight + 1, size=n)


def get_fitness(arr_binary: np.ndarray, arr_price: np.ndarray, arr_weight: np.ndarray,
                knapsack_capacity: int) -> int:
    """[Fitness function] Получает приспособленность, который представляет собой общую ценность предметов,
    помещенных в рюкзак. Если общая объем больше, чем объем рюкзака, вызовите функцию штрафа, чтобы установить
    приспособленность на отрицательное число.

    Args:
        arr_binary (np.ndarray): Массив после бинаризации (группы частиц)
        arr_price (np.ndarray): Массив ценностей
        arr_weight (np.ndarray): Массив весов
        knapsack_capacity (int): Объем рюкзака

    Returns:
        int: Приспособленность
    """
    if arr_binary.shape != arr_price.shape or arr_price.shape != arr_weight.shape:
        print('[ERROR] arr_binary, arr_price, arr_weight should have same shape')
        exit

    sum_price = 0
    sum_weight = 0
    for i in range(arr_binary.shape[0]):
        if 0 == arr_binary[i]:
            continue
        sum_price += arr_price[i]
        sum_weight += arr_weight[i]

    if sum_weight > knapsack_capacity:
        sum_price = -sum_price

    return sum_price


def get_index_max_price(arr_price: np.ndarray) -> int:
    """Получить подстрочный индекс элемента с наибольшим значением

    Args:
        arr_price (np.ndarray): Массивы для хранения стоимости предметов

    Returns:
        int: Подстрочный индекс для элементов с максимальным значением
    """
    return arr_price.tolist().index(arr_price.max())


def get_index_min_price(arr_price: np.ndarray) -> int:
    """Получить подстрочный индекс предмета с минимальным значением

    Args:
        arr_price (np.ndarray): Массивы для хранения ценности предметов

    Returns:
        int: индекс предмета с минимальным значением
    """
    return arr_price.tolist().index(arr_price.mix())


def get_index_max_cost_performance(arr_binary: np.ndarray, arr_price: np.ndarray, arr_weight: np.ndarray) -> int:
    """Вернуть подсписок самого большого производительности предмета, которого нет в рюкзаке
    производительность：p_i/w_i

    Args:
        arr_binary (np.ndarray): Массив после бинаризации (одномерные)
        arr_price (np.ndarray): Массив ценностей
        arr_weight (np.ndarray): Массив весов

    Returns:
        int: Абонементы для наиболее производительности предметов, не входящих в рюкзак
    """
    if arr_binary.shape != arr_price.shape or arr_price.shape != arr_weight.shape:
        print('[ERROR] arr_binary, arr_price, arr_weight should have same shape')
        exit

    index_max_cp = None
    max_cp = None
    for i in range(arr_binary.shape[0]):
        if 1 == arr_binary[i]:
            continue

        current_cp = arr_price[i] / arr_weight[i]
        if max_cp is None or current_cp > max_cp:
            max_cp = current_cp
            index_max_cp = i

    return index_max_cp


def get_index_min_cost_performance(arr_binary: np.ndarray, arr_price: np.ndarray, arr_weight: np.ndarray) -> int:
    """Верните подписку наименее производительности предмета в вашем рюкзаке
    производительность：p_i/w_i

    Args:
        arr_binary (np.ndarray): Массив после бинаризации (одномерные)
        arr_price (np.ndarray): Массив ценностей
        arr_weight (np.ndarray): Массив весов

    Returns:
        int: Подпись для наименее производительности предмета в рюкзаке
    """
    if arr_binary.shape != arr_price.shape or arr_price.shape != arr_weight.shape:
        print('[ERROR] arr_binary, arr_price, arr_weight should have same shape')
        exit

    index_min_cp = None
    min_cp = None
    for i in range(arr_binary.shape[0]):
        if 0 == arr_binary[i]:
            continue

        current_cp = arr_price[i] / arr_weight[i]
        if min_cp is None or current_cp < min_cp:
            min_cp = current_cp
            index_min_cp = i

    return index_min_cp


def repari_alg(arr_binary: np.ndarray, arr_price: np.ndarray, arr_weight: np.ndarray,
               knapsack_capacity: int) -> np.ndarray:
    """Алгоритм исправления  (RA, Repair Algorithm), который служит для исправления невыполнимых решений, возвращаемых
    алгоритмом PF (штрафная функция). и "вернуть" восстановленный новый массив

    Args:
        arr_binary (np.ndarray): Массив после бинаризации (одномерные)
        arr_price (np.ndarray): Массив ценностей
        arr_weight (np.ndarray): Массив весов
        knapsack_capacity (int): Объем рюкзака

    Returns:
        np.ndarray: Новый массив после восстановления алгоритма RA (одномерный массив)
    """
    arr_binary = copy(arr_binary)
    while get_fitness(arr_binary, arr_price, arr_weight, knapsack_capacity) < 0:
        index_min_cp = get_index_min_cost_performance(arr_binary, arr_price, arr_weight)
        arr_binary[index_min_cp] = 0

    return arr_binary


def improvement_alg(arr_binary: np.ndarray, arr_price: np.ndarray, arr_weight: np.ndarray,
                    knapsack_capacity: int) -> np.ndarray:
    """Алгоритм улучшения (IA, Improvement Algorithm), который служит для улучшения выполнимого решения,
    возвращаемого RA (алгоритмом исправления). и "вернуть" новый улучшенный массив Args: arr_binary (np.ndarray):
    Массив после бинаризации (одномерные) arr_price (np.ndarray): Массив ценностей arr_weight (np.ndarray): Массив
    весов knapsack_capacity (int): Объем рюкзака Returns: np.ndarray: Новый массив после улучшения алгоритма IM (
    одномерные массивы)
    """
    arr_binary = copy(arr_binary)
    last_array = copy(arr_binary)  # Массив перед временными улучшениями
    while get_fitness(arr_binary, arr_price, arr_weight, knapsack_capacity) > 0:
        last_array = copy(arr_binary)
        index_max_cp = get_index_max_cost_performance(arr_binary, arr_price, arr_weight)
        arr_binary[index_max_cp] = 1

    return last_array


def printSV():
    print('[INFO] -------------- V-Shape --------------')
    x = np.linspace(start=-10.0, stop=10.0, num=1000)
    plt.figure()
    for tf in [TransferFuncion.V1, TransferFuncion.V2, TransferFuncion.V3, TransferFuncion.V4]:
        y = [transfer_function(tf, a) for a in x]
        plt.plot(x, y, linestyle='-.')
    plt.title('Transfer Function\nV-Shape')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(['V1', 'V2', 'V3', 'V4'], loc='lower right')
    plt.show()

    print('[INFO] -------------- S-Shape --------------')
    plt.figure()
    for tf in [TransferFuncion.S1, TransferFuncion.S2, TransferFuncion.S3, TransferFuncion.S4]:
        y = [transfer_function(tf, a) for a in x]
        plt.plot(x, y, linestyle='-.')
    plt.title('Transfer Function\nS-Shape')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(['S1', 'S2', 'S3', 'S4'], loc='lower right')
    plt.show()


def BiEO(tf: TransferFuncion, arr_price: np.ndarray, arr_weight: np.ndarray, knapsack_capacity: int,
         num_groups_particle: int, max_iters: int, a_1: int, a_2: int, GP: float) -> np.ndarray:

    """
    Алгоритм оптимизации бинарного равновесия
    :param tf: Передаточная функция
    :param arr_price:  Массив ценностей
    :param arr_weight: Массив весов
    :param knapsack_capacity: Объем рюкзака
    :param num_groups_particle: Количество роев частиц
    :param max_iters: Максимальное количество итераций
    :param a_1: Способность к исследованию
    :param a_2: Способность к эксплуатации
    :param GP: Коэффициент генерации
    :return: Получен конечный результат равновесный бассейн и кандидаты (Equilibrium pool and candidates)
    """

    print('\n[INFO] -------------- Инициализация переменных --------------')
    num_particles_every_group = arr_weight.shape[0]  # (dim)
    print(f'Инициализация рюкзака：\n'
          f'\tОбъем рюкзака: {knapsack_capacity}\n'
          f'\tКоличество групп частиц: {num_groups_particle}\n'
          f'\tКоличество предметов в каждой группе (размерность): {num_particles_every_group}\n')

    print('\n[INFO] -------------- Массив ценностей --------------')
    print(f'{arr_price}')

    print('\n[INFO] -------------- Массив весов --------------')
    print(f'{arr_weight}')

    print(f'Инициализация параметров BiEO：\n'
          f'\tМаксимальное количество итераций: {max_iters}\n'
          f'\ta_1: {a_1}\n'
          f'\ta_2: {a_2}\n'
          f'\tПередаточная функция: {tf}\n'
          f'\tGP: {GP}')

    print('\n[INFO] -------------- Инициализация роя частиц C --------------')
    C = initialization(num_groups_particle, num_particles_every_group)  #  значение 0-1, помещается ли предмет в рюкзак
    print(f'C: {C}')

    print('\n[INFO] -------------- Инициализация роя частиц равновесного бассейна Ceq_1 ~ Ceq_4 --------------')
    Ceq_1 = np.zeros(num_particles_every_group)
    Ceq_1_fit = float('-inf')
    Ceq_2 = np.zeros(num_particles_every_group)
    Ceq_2_fit = float('-inf')
    Ceq_3 = np.zeros(num_particles_every_group)
    Ceq_3_fit = float('-inf')
    Ceq_4 = np.zeros(num_particles_every_group)
    Ceq_4_fit = float('-inf')

    print(f'\tCeq_1_fit: {Ceq_1_fit}\n'
          f'\tCeq_1: {Ceq_1}\n'
          f'\tCeq_1_fit: {Ceq_2_fit}\n'
          f'\tCeq_1: {Ceq_2}\n'
          f'\tCeq_1_fit: {Ceq_3_fit}\n'
          f'\tCeq_1: {Ceq_3}\n'
          f'\tCeq_1_fit: {Ceq_4_fit}\n'
          f'\tCeq_1: {Ceq_4}\n'
          )

    print('=============================== BiEO Начало итерации ===============================')
    it = 1
    C_pool = None
    while it <= max_iters:
        print(f'\n[INFO] -------------------- Текущая итерация {it}/{max_iters} --------------------')

        for i in range(C.shape[0]):
            C[i] = arr2bin(arr=C[i], tf=tf)
            fitness = get_fitness(arr_binary=C[i], arr_price=arr_price, arr_weight=arr_weight,
                                  knapsack_capacity=knapsack_capacity)
            print(f'\n[INFO] Текущая приспособленность роя частиц：{fitness}')

            if fitness < 0:
                C[i] = repari_alg(arr_binary=C[i], arr_price=arr_price, arr_weight=arr_weight,
                                  knapsack_capacity=knapsack_capacity)
                C[i] = improvement_alg(arr_binary=C[i], arr_price=arr_price, arr_weight=arr_weight,
                                       knapsack_capacity=knapsack_capacity)
                fitness = get_fitness(arr_binary=C[i], arr_price=arr_price, arr_weight=arr_weight,
                                      knapsack_capacity=knapsack_capacity)
                print(f'[INFO] Приспособленность после обращения в RA и IA：{fitness}')

            if fitness > Ceq_1_fit:
                # print('[INFO] Update Ceq_1')
                Ceq_1_fit = fitness
                Ceq_1 = copy(C[i])
            elif (fitness < Ceq_1_fit) and (fitness > Ceq_2_fit):
                # print('[INFO] Update Ceq_2')
                Ceq_2_fit = fitness
                Ceq_2 = copy(C[i])
            elif (fitness < Ceq_1_fit) and (fitness < Ceq_2_fit) and (fitness > Ceq_3_fit):
                # print('[INFO] Update Ceq_3')
                Ceq_3_fit = fitness
                Ceq_3 = copy(C[i])
            elif (fitness < Ceq_1_fit) and (fitness < Ceq_2_fit) and (fitness < Ceq_3_fit) and (fitness > Ceq_4_fit):
                # print('[INFO] Update Ceq_4')
                Ceq_4_fit = fitness
                Ceq_4 = copy(C[i])
            else:
                # print('[INFO] Не обновлено Ceq')
                pass
            print(f'[INFO] Текущая приспособленность Ceq1 ~ Ceq4: \n'
                  f'\tCeq_1_fit: {Ceq_1_fit}\n'
                  f'\tCeq_2_fit: {Ceq_2_fit}\n'
                  f'\tCeq_3_fit: {Ceq_3_fit}\n'
                  f'\tCeq_4_fit: {Ceq_4_fit}\n')

        Ceq_ave = np.round((Ceq_1 + Ceq_2 + Ceq_3 + Ceq_4) / 4)  # Среднее значение кандидатов в равновесном бассейне
        C_pool = np.array([Ceq_1, Ceq_2, Ceq_3, Ceq_4, Ceq_ave])  # Равновесный бассейн
        print(f'[INFO] Текущий равновесный бассейн: \n'
              f'\tCeq_1: {C_pool[0]}\t|\tCeq_1_fit: {Ceq_1_fit}\n'
              f'\tCeq_2: {C_pool[1]}\t|\tCeq_2_fit: {Ceq_2_fit}\n'
              f'\tCeq_3: {C_pool[2]}\t|\tCeq_3_fit: {Ceq_3_fit}\n'
              f'\tCeq_4: {C_pool[3]}\t|\tCeq_4_fit: {Ceq_4_fit}\n'
              f'\tCeq_ave: {C_pool[4]}\n')

        t = (1 - it / max_iters) ** (a_2 * it / max_iters)  # Eq(4)
        for i in range(C.shape[0]):
            Ceq = C_pool[np.random.randint(C_pool.shape[0])]  # Один случайный выбор из равновесного бассейна
            lambda_F = np.random.random(num_particles_every_group)
            r = np.random.random(num_particles_every_group)
            F = a_1 * np.sign(r - 0.5) * (np.exp(-lambda_F * t) - 1)  # Eq(6)
            GCP = 0.5 * np.random.random() * np.ones(num_particles_every_group) * (np.random.random() > GP)  # Eq(9)
            G_0 = GCP * (Ceq - lambda_F * C[i])  # Eq(8)
            G = G_0 * F  # Eq(7)
            C[i] = Ceq + (C[i] - Ceq) * F + (G / lambda_F) * (1 - F)  # Eq(10)

        it += 1

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Конец итерации @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f'4 оптимальных решения：\n'
          f'\tCeq_1: {C_pool[0]}\t|\tCeq_1_fit: {Ceq_1_fit}\n'
          f'\tCeq_2: {C_pool[1]}\t|\tCeq_2_fit: {Ceq_2_fit}\n'
          f'\tCeq_3: {C_pool[2]}\t|\tCeq_3_fit: {Ceq_3_fit}\n'
          f'\tCeq_4: {C_pool[3]}\t|\tCeq_4_fit: {Ceq_4_fit}\n'
          f'\tCeq_avg_fit: {(Ceq_1_fit + Ceq_2_fit + Ceq_3_fit + Ceq_4_fit) / 4}\n')

    return C_pool


if __name__ == '__main__':
    print('==================== main start ====================')

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Установите параметры здесь @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    arr_tf = [TransferFuncion.S1, TransferFuncion.S2, TransferFuncion.S3, TransferFuncion.S4,
              TransferFuncion.V1, TransferFuncion.V2, TransferFuncion.V3, TransferFuncion.V4]
    arr_kp_c = [KP_1_c, KP_2_c, KP_3_c, KP_4_c, KP_5_c, KP_6_c, KP_7_c, KP_8_c, KP_9_c, KP_10_c,
                KP_11_c, KP_12_c, KP_13_c, KP_14_c, KP_15_c, KP_16_c, KP_17_c, KP_18_c, KP_19_c, KP_20_c]
    arr_w = [KP_1_w, KP_2_w, KP_3_w, KP_4_w, KP_5_w, KP_6_w, KP_7_w, KP_8_w, KP_9_w, KP_10_w,
             KP_11_w, KP_12_w, KP_13_w, KP_14_w, KP_15_w, KP_16_w, KP_17_w, KP_18_w, KP_19_w, KP_20_w]
    arr_v = [KP_1_v, KP_2_v, KP_3_v, KP_4_v, KP_5_v, KP_6_v, KP_7_v, KP_8_v, KP_9_v, KP_10_v,
             KP_11_v, KP_12_v, KP_13_v, KP_14_v, KP_15_v, KP_16_v, KP_17_v, KP_18_v, KP_19_v, KP_20_v]

    cur_res = np.zeros(shape=(20, 8))
    arr_res = np.zeros(shape=(20, 8))
    for num_run in range(1):  # runs
        for i in range(20):
            # print(f'\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%% i: {i} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('\n')

            for j in range(8):
                """Вызов алгоритма BiEO"""
                blockPrint()
                C_pool = BiEO(tf=arr_tf[j],
                              arr_price=arr_v[i],
                              arr_weight=arr_w[i],
                              knapsack_capacity=arr_kp_c[i],
                              num_groups_particle=20,  # [20]
                              max_iters=5000,  # [5000]
                              a_1=3,  # [3]
                              a_2=1,  # [1]
                              GP=0.5)
                enablePrint()

                # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Конец итерации @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                cur_best_fitness = round(get_fitness(C_pool[0], arr_v[i], arr_w[i], arr_kp_c[i]))
                cur_res[i, j] = cur_best_fitness
                print(f'[INFO] run:{num_run} Конец итерации (Набор данных：KP_{i + 1}\tПередаточная функция：{arr_tf[j]})-Оптимальное решение Ceq_avg_fit: {cur_best_fitness}')
        arr_res = np.maximum(arr_res, cur_res)

    print(f'result:\n{arr_res}')
