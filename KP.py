# ------*------ coding: utf-8 ------*------
# @Time    : 2022/11/9 21:14
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Binary-Equilibrium-OA
# @File    : KP.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import numpy as np

KP_1_c = 269
KP_1_w = np.array([95, 4, 60, 32, 23, 72, 80, 62, 65, 46])
KP_1_v = np.array([55, 10, 47, 5, 4, 50, 8, 61, 85, 87])

KP_2_c = 878
KP_2_w = np.array([92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70, 48, 14, 58])
KP_2_v = np.array([44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 7, 15, 61, 17, 75, 29, 75, 63])

KP_3_c = 20
KP_3_w = np.array([6, 5, 9, 7])
KP_3_v = np.array([9, 11, 13, 15])

KP_4_c = 11
KP_4_w = np.array([2, 4, 6, 7])
KP_4_v = np.array([6, 10, 12, 13])

KP_5_c = 375
KP_5_w = np.array(
    [56.358531, 80.874050, 47.987304, 89.596240, 74.660482, 85.894345, 51.353496, 1.498459, 36.445204, 16.589862,
     44.569231, 0.466933, 37.788018, 57.118442, 60.716575])
KP_5_v = np.array(
    [0.125126, 19.330424, 58.500931, 35.029145, 82.284005, 17.410810, 71.050142, 30.399487, 9.140294, 14.731285,
     98.852504, 11.908322, 0.891140, 53.166295, 60.176397])

KP_6_c = 60
KP_6_w = np.array([30, 25, 20, 18, 17, 11, 5, 2, 1, 1])
KP_6_v = np.array([20, 18, 17, 15, 15, 10, 5, 3, 1, 1])

KP_7_c = 50
KP_7_w = np.array([31, 10, 20, 19, 4, 3, 6])
KP_7_v = np.array([70, 20, 39, 37, 7, 5, 10])

KP_8_c = 10000
KP_8_w = np.array(
    [983, 982, 981, 980, 979, 978, 488, 976, 972, 486, 486, 972, 972, 485, 485, 969, 966, 483, 964, 963, 961, 958, 959])
KP_8_v = np.array(
    [981, 980, 979, 978, 977, 976, 487, 974, 970, 485, 485, 970, 970, 484, 484, 976, 974, 482, 962, 961, 959, 958, 857])

KP_9_c = 80
KP_9_w = np.array([15, 20, 17, 8, 31])
KP_9_v = np.array([33, 24, 36, 37, 12])

KP_10_c = 879
KP_10_w = np.array([84, 83, 43, 4, 44, 6, 82, 92, 25, 83, 56, 18, 58, 14, 48, 70, 96, 32, 68, 92])
KP_10_v = np.array([91, 72, 90, 46, 55, 8, 35, 75, 61, 15, 77, 40, 63, 75, 29, 75, 17, 78, 40, 44])

KP_11_c = 577
KP_11_w = np.array(
    [46, 17, 35, 1, 26, 17, 17, 48, 38, 17, 32, 21, 29, 48, 31, 8, 42, 37, 6, 9, 15, 22, 27, 14, 42, 40, 14, 31, 6, 34])
KP_11_v = np.array(
    [57, 64, 50, 6, 52, 6, 85, 60, 70, 65, 63, 96, 18, 48, 85, 50, 77, 18, 70, 92, 17, 43, 5, 23, 67, 88, 35, 3, 91,
     48])

KP_12_c = 655
KP_12_w = np.array(
    [7, 4, 36, 47, 6, 33, 8, 35, 32, 3, 40, 50, 22, 18, 3, 12, 30, 31, 13, 33, 4, 48, 5, 17, 33, 26, 27, 19, 39, 15, 33,
     47, 17, 41, 40])
KP_12_v = np.array(
    [35, 67, 30, 69, 40, 40, 21, 73, 82, 93, 52, 20, 61, 20, 42, 86, 43, 93, 38, 70, 59, 11, 42, 93, 6, 39, 25, 23, 36,
     93, 51, 81, 36, 46, 96])

KP_13_c = 819
KP_13_w = np.array(
    [28, 23, 35, 38, 20, 29, 11, 48, 26, 14, 12, 48, 35, 36, 33, 39, 30, 26, 44, 20, 13, 15, 46, 36, 43, 19, 32, 2, 47,
     24, 26, 39, 17, 32, 17, 16, 33, 22, 6, 12])
KP_13_v = np.array(
    [13, 16, 42, 69, 66, 68, 1, 13, 77, 85, 75, 95, 92, 23, 51, 79, 53, 62, 56, 74, 7, 50, 23, 34, 56, 75, 42, 51, 13,
     22, 30, 45, 25, 27, 90, 59, 94, 62, 26, 11])

KP_14_c = 907
KP_14_w = np.array(
    [18, 12, 38, 12, 23, 13, 18, 46, 1, 7, 20, 43, 11, 47, 49, 19, 50, 7, 39, 29, 32, 25, 12, 8, 32, 41, 34, 24, 48, 30,
     12, 35, 17, 38, 50, 14, 47, 35, 5, 13, 47, 24, 45, 39, 1])
KP_14_v = np.array(
    [98, 70, 66, 33, 2, 58, 4, 27, 20, 45, 77, 63, 32, 30, 8, 18, 73, 9, 92, 43, 8, 58, 84, 35, 78, 71, 60, 38, 40, 43,
     43, 22, 50, 4, 57, 5, 88, 87, 34, 98, 96, 99, 16, 1, 25])

KP_15_c = 882
KP_15_w = np.array(
    [15, 40, 22, 28, 50, 35, 49, 5, 45, 3, 7, 32, 19, 16, 40, 16, 31, 24, 15, 42, 29, 4, 14, 9, 29, 11, 25, 37, 48, 39,
     5, 47, 49, 31, 48, 17, 46, 1, 25, 8, 16, 9, 30, 33, 18, 3, 3, 3, 4, 1])
KP_15_v = np.array(
    [78, 69, 87, 59, 63, 12, 22, 4, 45, 33, 29, 50, 19, 94, 95, 60, 1, 91, 69, 8, 100, 70, 84, 100, 32, 81, 47, 59, 48,
     56, 18,
     59, 16, 45, 54, 4798, 75, 20, 4, 19, 58, 63, 37, 64, 90, 26, 29, 13, 53, 83])

KP_16_c = 1050
KP_16_w = np.array(
    [27, 15, 46, 5, 40, 9, 36, 12, 11, 11, 49, 20, 32, 3, 12, 44, 24, 1, 24, 42, 44, 16, 12, 42, 22, 26, 10, 8, 46, 50,
     20, 42, 48, 45, 43, 35, 9, 12, 22, 2, 14, 50, 16, 29, 31, 46, 20, 35, 11, 4, 32, 35, 15, 29, 16])
KP_16_v = np.array(
    [98, 74, 76, 4, 12, 27, 90, 98, 100, 35, 30, 19, 75, 72, 19, 44, 5, 66, 79, 87, 79, 44, 35, 6, 82, 11, 1, 28, 95,
     68, 39, 86, 68, 61, 44, 97, 83, 2, 15, 49, 59, 30, 44, 40, 14, 96, 37, 84, 5, 43, 8, 32, 95, 86, 18])

KP_17_c = 1006
KP_17_w = np.array(
    [7, 13, 47, 33, 38, 41, 3, 21, 37, 7, 32, 13, 42, 42, 23, 20, 49, 1, 20, 25, 31, 4, 8, 33, 11, 6, 3, 9, 26, 44, 39,
     7, 4, 34, 25, 25, 16, 17, 46, 23, 38, 10, 5, 11, 28, 34, 47, 3, 9, 22, 17, 5, 41, 20, 33, 29, 1, 33, 16, 14])
KP_17_v = np.array(
    [81, 37, 70, 64, 97, 21, 60, 9, 55, 85, 5, 33, 71, 87, 51, 100, 43, 27, 48, 17, 16, 27, 76, 61, 97, 78, 58, 46, 29,
     76, 10, 11, 74, 36, 59, 30, 72, 37, 72, 100, 9, 47, 10, 73, 92, 9, 52, 56, 69, 30, 61, 20, 66, 70, 46, 16, 43, 60,
     33, 84])

KP_18_c = 1319
KP_18_w = np.array(
    [47, 27, 24, 27, 17, 17, 50, 24, 38, 34, 40, 14, 15, 36, 10, 42, 9, 48, 37, 7, 43, 47, 29, 20, 23, 36, 14, 2, 48,
     50, 39, 50, 25, 7, 24, 38, 34, 44, 38, 31, 14, 17, 42, 20, 5, 44, 22, 9, 1, 33, 19, 19, 23, 26, 16, 24, 1, 9, 16,
     38, 30, 36, 41, 43, 6])
KP_18_v = np.array(
    [47, 63, 81, 57, 3, 80, 28, 83, 69, 61, 39, 7, 100, 67, 23, 10, 25, 91, 22, 48, 91, 20, 45, 62, 60, 67, 27, 43, 80,
     94, 47, 31, 44, 31, 28, 14, 17, 50, 9, 93, 15, 17, 72, 68, 36, 10, 1, 38, 79, 45, 10, 81, 66, 46, 54, 53, 63, 65,
     20, 81, 20, 42, 24, 28, 1])

KP_19_c = 1426
KP_19_w = np.array(
    [4, 16, 16, 2, 9, 44, 33, 43, 14, 45, 11, 49, 21, 12, 41, 19, 26, 38, 42, 20, 5, 14, 40, 47, 29, 47, 30, 50, 39, 10,
     26, 33, 44, 31,
     50, 7, 15, 24, 7, 12, 10, 34, 17, 40, 28, 12, 35, 3, 29, 50, 19, 28, 47, 13, 42, 9, 44, 14, 43, 41, 10, 49, 13, 39,
     41, 25, 46, 6, 7, 43])
KP_19_v = np.array(
    [66, 76, 71, 61, 4, 20, 34, 65, 22, 8, 99, 21, 99, 62, 25, 52, 72, 26, 12, 55, 22, 32, 98, 31, 95, 42, 2, 32, 16,
     100, 46, 55, 27, 89, 11, 8, 3, 43, 93, 53, 88, 36, 41, 60, 92, 14, 5, 41, 60, 92, 30, 55, 79, 33, 10, 45, 3, 68,
     12, 20, 54, 63, 38, 61, 85, 71, 40, 58, 25, 73])

KP_20_c = 1433
KP_20_w = np.array(
    [24, 45, 15, 40, 9, 37, 13, 5, 43, 35, 48, 50, 27, 46, 24, 45, 2, 7, 38, 20, 20, 31, 2, 20, 3, 35, 27, 4, 21, 22,
     33, 11, 5, 24, 37, 31, 46, 13, 12, 12, 41, 36, 44, 36, 34, 22, 29, 50, 48, 17, 8, 21, 28, 2, 44, 45, 25, 11, 37,
     35, 24, 9, 40, 45, 8, 47, 1, 22, 1, 12, 36, 35, 14, 17, 5])
KP_20_v = np.array(
    [2, 73, 82, 12, 49, 35, 78, 29, 83, 18, 87, 93, 20, 6, 55, 1, 83, 91, 71, 25, 59, 94, 90, 61, 80, 84, 57, 1, 26, 44,
     44, 88, 7, 34, 18, 25, 73, 29, 24, 14, 23, 82, 38, 67, 94, 43, 61, 97, 37, 67, 32, 89, 30, 30, 91, 50, 21, 3, 18,
     31, 97, 79, 68, 85, 43, 71, 49, 83, 44, 86, 1, 100, 28, 4, 16])
