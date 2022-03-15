# Subgradient Method
This repository is meant for a project I did in the course AI2101 Convex Optimization at IIT-H spring 2022 on the topic subgradient method.
## Motivation
In practice, many functions can be non-differentiable at certain places in their domain, which means that the traditional gradient descent methods don't work any longer since they work under the assumptions that the function is differentiable. Subgradient method is an iterative first-order method which is same as gradient descent at places where the gradient is defined, but at the places where the function is non-differentiable, it uses minimizer to get a range of gradients, all of which are subgradients.