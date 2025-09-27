# MachineLearning_Assignment1: The Three-Body Problem

## Overview

This repository focuses on solving the famous **Three-Body Problem** using machine learning techniques. The Three-Body Problem is a classical problem in physics and celestial mechanics that involves predicting the motion of three celestial bodies moving under the influence of their mutual gravitational attraction.

## The Three-Body Problem

The Three-Body Problem is a special case of the n-body problem in classical mechanics. Unlike the two-body problem, which has a closed-form analytical solution, the three-body problem generally does not have a simple analytical solution and exhibits chaotic behavior for most initial conditions.

## Tasks

The primary objective of this project is to:

**Predict the movement of the three bodies on a 2D plane given a set of initial positions, where at the initial positions, velocities are zero.**

### Dataset Description

We have access to a dataset from simulations of the 3-Body Problem under specific initial conditions. The dataset contains:

- **Position measurements**: x₁, y₁ for each body
- **Velocity measurements**: vₓ₁, vᵧ₁ for each body
- **Time series data**: Multiple time steps showing the evolution of the system

### Data Structure

For each body (i = 1, 2, 3) at each time step t:
- Position coordinates: (xᵢ(t), yᵢ(t))
- Velocity components: (vₓᵢ(t), vᵧᵢ(t))

The challenge is to learn the underlying dynamics from this simulation data and make accurate predictions for new initial conditions.


