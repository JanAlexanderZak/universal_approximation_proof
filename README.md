# Universal Approximation Theorem: Experimental Proofs  

This repository contains a series of experimental implementations aimed at demonstrating and verifying the Universal Approximation Theorem. It states that a neural network with a single hidden layer and a sufficiently large number of neurons can approximate any continuous function on a closed interval, given appropriate activation functions.

## Absolute Value
The absolute value function is piecewise linear with a non-differentiable point at the origin:
```math
|x| = \left\{ \begin{array}{cl}
x & : \ x \geq 0 \\
-x & : \ x < 0
\end{array} \right.
```
![](https://github.com/JanAlexanderZak/universal_approximation_proof/blob/main/src/plots/abs.gif)

## Hyperbolic Tangent
The hyperbolic tangent is a smooth, bounded activation function mapping inputs to $(-1, 1)$:
```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```
![](https://github.com/JanAlexanderZak/universal_approximation_proof/blob/main/src/plots/tanh.gif)

## Heaviside Step Function
The Heaviside function is a discontinuous step function, making it a challenging approximation target:
```math
H(x) = \left\{ \begin{array}{cl}
1 & : \ x \geq 0 \\
0 & : \ x < 0
\end{array} \right.
```
![](https://github.com/JanAlexanderZak/universal_approximation_proof/blob/main/src/plots/heaviside.gif)

## 2D Surface: sin(x) * cos(y)
A 2D target function that tests the network's ability to approximate multivariate mappings:
```math
f(x, y) = \sin(x) \cdot \cos(y)
```
![](https://github.com/JanAlexanderZak/universal_approximation_proof/blob/main/src/plots/sin_cos_2d.gif)