# MachineLearning_Assignment1: The Three-Body Problem

## Overview

This repository focuses on solving the famous **Three-Body Problem** using machine learning techniques. The Three-Body Problem is a classical problem in physics and celestial mechanics that involves predicting the motion of three celestial bodies moving under the influence of their mutual gravitational attraction.

## The Three-Body Problem

The Three-Body Problem is a special case of the n-body problem in classical mechanics. Unlike the two-body problem, which has a closed-form analytical solution, the three-body problem generally does not have a simple analytical solution and exhibits chaotic behavior for most initial conditions.

### Mathematical Background

For three bodies with masses m₁, m₂, and m₃, the gravitational forces between them create a complex system of differential equations. Each body experiences gravitational forces from the other two bodies, resulting in highly nonlinear dynamics that can lead to:

- Chaotic trajectories
- Periodic orbits
- Escape scenarios
- Collision events

The system's evolution is highly sensitive to initial conditions, making it an excellent candidate for machine learning approaches.

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

### Expected Deliverables

1. **Data Analysis**: Explore and visualize the three-body simulation data
2. **Model Development**: Create machine learning models to predict body trajectories
3. **Evaluation**: Assess model performance on unseen initial conditions
4. **Visualization**: Generate plots showing predicted vs. actual trajectories

### Applications

Understanding and predicting three-body dynamics has applications in:
- Orbital mechanics and spacecraft trajectory planning
- Stellar system dynamics
- Asteroid and comet trajectory prediction
- Chaos theory and dynamical systems research

## Getting Started

1. Clone this repository
2. Install required dependencies (to be added)
3. Load and explore the three-body simulation dataset
4. Implement and train prediction models
5. Evaluate and visualize results

## Project Structure

```
MachineLearning_Assignment1/
├── README.md                 # This file
├── .gitignore               # Git ignore file for Python projects
├── data/                    # Dataset directory (to be added)
├── notebooks/               # Jupyter notebooks for analysis (to be added)
├── src/                     # Source code (to be added)
├── models/                  # Trained models (to be added)
└── results/                 # Output results and visualizations (to be added)
```

## Contributing

This is an educational project for machine learning assignment. Follow standard Python and Jupyter notebook development practices.
