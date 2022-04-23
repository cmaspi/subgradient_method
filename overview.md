# Introduction
Subgradient method is a simple algorithm used to minimize nondifferentiable convex functions. This method is similar to vanilla gradient method which is used for differentiable functions, except.
1. this method applied directly to nondifferentiable function f.
2. the step lengths are not chosen via line search and are fixed ahead of time (ignore this)
3. unlike the gradient method, this method is not a descent method, the function value and (and often does) increase.

# Limitations
They tend to be slower than interior point methods or newton's method in an unconstrained case, their performance depends on the scaling of the problem.
# advantages
they can be immediately applied to a far wider variety of problems, their memory requirement is far less

# basic subgradient method
start with the unconstrained case, where goal is to minimize f, which is convex

$x^{(k+1)} = x^{(k)} - \alpha_kg^{(k)}$
