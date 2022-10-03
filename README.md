<p align="center">
 <img width="100px" src="https://raw.githubusercontent.com/NekoSilverFox/NekoSilverfox/403ab045b7d9adeaaf8186c451af7243f5d8f46d/icons/silverfox.svg" align="center" alt="NekoSilverfox" />
 <p align="center"><b><font size=6>0-1背包问题的二元均衡优化算法</font></b></p>
 <p align="center"><b>A Binary Equilibrium Optimization Algorithm for 0–1 Knapsack Problems</b></p>
</p>



 <p align="center">
  <img width="250px" src="https://github.com/NekoSilverFox/NekoSilverfox/blob/master/icons/logo_building_spbstu.png?raw=true" align="center" alt="ogo_building_spbstu" />
  </br>
  </br>
  <b><font size=4>Санкт-Петербургский государственный политехнический университет</font></b></br>
  <b>Институт компьютерных наук и технологий</b>
 </p>
 <p align="center"></p>

<div align=left>
<div align=center>


[![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen)](LICENSE)



<div align=left>

[toc]

# 概述

> 算法来自于论文：https://doi.org/10.1016/j.cie.2020.106946
>
> ---
>
> **A Binary Equilibrium Optimization Algorithm for 0–1 Knapsack Problems**
>
> ---
>
> Mohamed Abdel-Basset, Reda Mohamed, Seyedali Mirjalilib
>
> Faculty of Computers and Informatics, Zagazig University, Zagazig 44519, Egypt
> Center for Artificial Intelligence Research and Optimization, Torrens University Australia, AustraliaYonsei Frontier Lab, Yonsei University, Seoul, Korea
> d *King Abdul Aziz University, Jeddah, Saudi Arabia*



该仓库将注重本算法的实现及验证



# 缩写说明

## 专有名词

| 缩写 | 说明                                                         |
| ---- | ------------------------------------------------------------ |
| KP   | Knapsack Problems - 背包问题                                 |
| EO   | equilibrium optimizer - 标准的均衡优化器                     |
| BEO  | binary version of equilibrium optimization - 二进制版本的均衡优化 |
| RA   | repair algorithm - 修复算法                                  |
|      |                                                              |
|      |                                                              |
|      |                                                              |
|      |                                                              |
|      |                                                              |



## 符号

| 符号    | 说明                          |
| ------- | ----------------------------- |
| $n$     | 0-1 背包问题中，一共有 n 项   |
| $i$     | 背包中的第 $i$ 项             |
| $w_i$   | 第 $i$ 项的权重（占用的空间） |
| $p_i$   | 第 $i$ 项的利润（价值）       |
| $x_i=1$ | 第 $i$ 项物品装入了背包       |
| $x_i=0$ | 第 $i$ 项没有装入背包         |
|         |                               |
|         |                               |
|         |                               |
|         |                               |



# 摘要

本文提出了一个二进制版本的均衡优化（BEO - binary version of equilibrium optimization），用于解决0-1 knapsack问题，该问题被描述为一个离散问题。**由于标准的[均衡优化器](https://zhuanlan.zhihu.com/p/418152665)（EO - equilibrium optimizer）是为解决连续优化问题而提出的，因此需要一个离散的变体来解决二进制问题**。因此，包括V型和S型在内的八个转移函数被用来将连续EO转换为二进制EO（BEO）。在这些传递函数中，本研究表明，V形V3是最好的函数。我们还发现，S3传输函数比V3更有利于提高本文所采用的其他算法的性能。我们得出结论，**任何二进制算法的性能都依赖于传递函数的良好选择**。此外，我们使用惩罚函数从问题的解决方案中筛选出不可行的解决方案，并应用修复算法（RA - repair algorithm）将其转换为可行的解决方案。我们在三个基准数据集上评估了所提出的算法的性能，这些数据集有63个小型、中型和大型实例，并在不同的统计分析下与其他一些为解决0-1 knapsack而提出的算法进行了比较。实验结果表明，BEOV3算法在所有的小型、中型案例研究中都有优势。关于大规模的测试案例，所提出的方法在18个实例中的13个达到了最优值。



# 简介（第1节）

> 背包问题简单来说就是，在固定的背包容量下，装入价值最多最高的物品

**Knapsack问题（KP）**在许多现实世界的应用中得到了它们的重要性。这些问题常见于投资决策（Rooderkerk和van Heerde，2016）、货物装载问题（Mladenović，2019，Cho，2019，Brandt和Nickel，2019）、能源最小化（Müller，2015，Karaboghossian和Zito，2018）、资源分配（Jacko，2016）、计算机内存（Oppong，2019）、项目组合选择（Koc，2009，Bas，2012，Tavana，2015，Tavana et al, 2013），自适应多媒体系统（Khan，2002），密码学（Khan，2002，Liu等人，2019），住房问题（Chan，2018），以及切割库存问题（Alfares和Alsawafy，2019）。因此，解决KP使许多应用得以蓬勃发展和壮大。然而，KP是一个NP-hard问题，所以找到一个多项式时间的解决方案是困难的。**KP是一个组合优化问题，我们寻求在有限的解决方案中找到最佳解决方案**。**如果KP的大小增加，精确方法搜索最优解所需的时间就会呈指数级增长**。精确的方法，如穷举搜索或分支和约束，在空间和时间方面消耗了巨大的计算资源，特别是对于大规模的KP。因此，需要在可接受的时间内找到一个接近最优的解决方案，这是许多研究人员在解决KP时遵循的一个趋势。

在0-1 KP中，假设给定了一个由**==n个项目==**组成的集合，每个项目都有一个私人==$权重 w_i$==和==$利润 p_i$==。从给定的n个项目中，**决策者需要一个能使利润最大化的子集，同时保持其权重之和小于或等于背包的容量**。为了找到这个项目子集，**如果背包包含所选的==第i项==，其值为$x_i=1$，如果没选入背包值为$x_i=0$**。最后，这个问题在数学上可以表述为：
$$
\operatorname{maximize} \sum_{i=1}^{n} x_{i}^{*} p_{i}

\\

subjectto (使)\sum_{i=1}^{n} w_{i}^{*} x_{i}<c

\\

x_{i}=0 \ or \ 1, i=0,1 \cdots \cdots n

\\

p_{i}>0,\ w_{i}>0,\ c>0
$$
最近，许多**[元启发式算法](https://zh.wikipedia.org/wiki/元启发算法)（*meta*-heuristic algorithms）**被建议为各种优化问题寻找最优解，并取得了相对较好的结果（Bairathi and Gopalani, 2018; Mirjalili and Lewis, 2016; Askarzadeh, 2016; Abualigah, 2019; Abualigah et al., 2018; Abualigah et al., 2018; Mohammad Abualigah, 2020; Abualigah and Diabat, 2020; Safaldin et al., 2020）。**与精确方法相比，元启发式方法的特点是更快地收敛到最优解，并减少了计算成本**。因此，许多作者争相使用元启发法来解决KP。任何元启发式算法的主要目标是有效地探索搜索空间，找到接近最优的解决方案。一个稳健的元启发式算法是能够保持探索和利用阶段之间的平衡。下一节将讨论一些为解决0-1结包而提出的元启发式算法。

元启发式算法的优异表现促使我们提出了一种新的元启发式算法的二进制版本，即均衡优化器（EO）（Faramarzi，2019），其灵感来自物理学，用于处理knapsack问题。该算法在处理连续问题方面的高稳定性是提出二进制版本的原因，以研究其在处理作为组合优化问题的0-1 knapsack问题时的性能。元启发式算法相对于现有算法的优势总结如下：

1. 避免过早向局部最优收敛
2. 在迭代结束前有很强的稳定性
3. 有两个因子帮助算法平衡和前进
4. 算法存储目前为止最好的 4 个解，使算法具有额外的能力，以避免陷入局部最小值，从而加速向最优解决方案收敛。

根据这些优点，它被认为是一个强大的算法，随后调查它在离散问题上的表现是一个不可缺少的命令。**为了将EO的连续值转换为离散值，使用了八个转移函数，即V-Shape和S-Shape，并对这些函数进行了广泛的实验，以获得性能最好的一个二进制版本的EO（BEO）**。经过实验，很明显，**V-Shape V3是最好的一个**。此外，为了检查BEO的性能，它在小型、中型和大型三个基准数据集上进行了验证，并与14种最先进的算法进行了比较。经过验证和比较，所提出的算法在三个基准的大多数实例上，**特别是在大规模的实例上的优势是非常明显的**。最后，本文中主要贡献有以下几点：

- 提出了解决0-1结包问题的二元版新型EO
- **增加一个决策模型，在解决大规模的背包问题上有更高的能力**，为剪切原料寻找最不丰富的方式，选择投资和组合，为默克尔-海尔曼(Lagarias, 1984)生成关键和解决其他背包问题(Kellerer et al.， 2004)
- **提出了转移函数的良好选择可以提高二进制算法的性能**



本文的其余部分组织如下：

第2节：之前解决 0-1KP 和 MKP 的工作

第3节：总结了原始的EO。

第4节：说明了使用充分的转移函数作为建议，说明了 EO 在解决 0-1KP 的适应性。

第5节：介绍了所提出的方法在三组标准的知名基准上解决0-1 knapsack问题的讨论和实验结果，并进行了详细说明。

第6节：提供了关于建议的方法和未来工作的一些结论。



# 文献回顾（第2节）





























