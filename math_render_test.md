# 公式渲染测试矩阵

## 1. 多行 $$ 带 \tag（当前写法）

$$
E_{eff}\frac{dSOC}{dt}=-P_{tot} \tag{1}
$$

## 2. 多行 $$ 不带 \tag

$$
E_{eff}\frac{dSOC}{dt}=-P_{tot}
$$

## 3. 单行 $$ 带 \tag

$$E_{eff}\frac{dSOC}{dt}=-P_{tot} \tag{3}$$

## 4. 单行 $$ 不带 \tag

$$E_{eff}\frac{dSOC}{dt}=-P_{tot}$$

## 5. ```math 带 \tag

```math
E_{eff}\frac{dSOC}{dt}=-P_{tot} \tag{5}
```

## 6. ```math 不带 \tag

```math
E_{eff}\frac{dSOC}{dt}=-P_{tot}
```

## 7. 多行 $$ 带 cases

$$
u(T)=\begin{cases}1, & T\le T_0\\ u_{min}+(1-u_{min})e^{-\frac{T-T_0}{\tau}}, & T>T_0\end{cases} \tag{7}
$$

## 8. ```math 带 cases

```math
u(T)=\begin{cases}1, & T\le T_0\\ u_{min}+(1-u_{min})e^{-\frac{T-T_0}{\tau}}, & T>T_0\end{cases} \tag{8}
```

## 9. 行内公式测试

行内 $E_{eff}$ 与 $P_{tot}$ 以及 $k_0$ 测试。
