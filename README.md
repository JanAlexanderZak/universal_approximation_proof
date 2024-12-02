# Universal Approximation Theorem: Experimental Proofs  

This repository contains a series of experimental implementations aimed at demonstrating and verifying the Universal Approximation Theorem. It states that a neural network with a single hidden layer and a sufficiently large number of neurons can approximate any continuous function on a closed interval, given appropriate activation functions.

For example, the absolute value function in the form of:
```math
|x| = \left\{ \begin{array}{cl}
x & : \ x \geq 0 \\
-x & : \ x < 0
\end{array} \right. \, .
```
can be approximated. The process of learning the function over a number of epochs can be visualized:

![](https://github.com/JanAlexanderZak/universal_approximation_proof/blob/main/src/plots/abs.gif)  