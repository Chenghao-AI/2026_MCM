# 修复方案验证

## A. 编号在右（\text 替代 \tag）

$$
E_{eff}\frac{dSOC}{dt}=-P_{tot}\qquad\text{(1)}
$$

## B. cases + 编号在右

$$
u(T)=\begin{cases}1, & T\le T_0\\ u_{min}+(1-u_{min})e^{-\frac{T-T_0}{\tau}}, & T>T_0\end{cases}\qquad\text{(6)}
$$

## C. 两个公式并排（4-5 式）

$$
f=f_0\,u(T),\qquad f_{dis}=60\,u(T)\qquad\text{(4-5)}
$$

## D. 长公式（温度方程）

$$
C\frac{dT}{dt}=(1-\eta_0)P_{CPU}+(1-\eta_n)P_{net}+(1-\eta_d)P_{dis}-\lambda S(T-T_0)\qquad\text{(3)}
$$

## E. 积分公式（老化）

$$
\frac{\Delta E_{eff}}{E_{eff}}=1-e^{-\int_0^T k\,dt}\qquad\text{(17)}
$$
