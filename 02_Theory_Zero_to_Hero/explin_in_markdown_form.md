# 02_Theory_Zero_to_Hero: Evolution Strategies Demystified

This module builds a complete theoretical and intuitive foundation for **"Evolution Strategies as a Scalable Alternative to Reinforcement Learning"** (Salimans et al., OpenAI, 2017). Starting from basic neural network mechanics, every mathematical equation, biological analogy, and systems engineering trick in the paper is derived step by step without skipped transitions.

## Module 1: Level Zero — Inputs, Brain, Environment, and Return

In Reinforcement Learning (RL) and black-box control, the goal is to build an **Agent** that interacts with an **Environment** (such as an Atari video game or a MuJoCo physics simulator) to master a task.

### 1.1 The Interaction Loop

At each discrete time step $t$ (from the start of an episode at $t = 1$ until termination at $t = T$), three operations occur in sequence:

1. **Observation / State ($s_t$):** The environment sends the current situation to the agent, such as raw pixel values from a screen or a vector of joint angles and velocities of a robot.
2. **Policy Network ($\pi_\theta$):** The agent's "brain" is a neural network parameterized by weights $\theta$. It takes state $s_t$ as input and computes an **Action ($a_t$):**

$$a_t = \pi(s_t; \theta)$$


3. **Environment Transition & Reward ($r_t$):** The action $a_t$ is applied to the environment, which advances the physics simulation to the next state $s_{t+1}$ and emits an immediate scalar reward $r_t$.

When the episode finishes at step $T$, all immediate rewards are summed into a single scalar score called the **Total Return** or **Fitness**, denoted by $F(\theta)$:

$$F(\theta) = \sum_{t=1}^{T} r_t$$

The entire optimization problem consists of finding the specific weight vector $\theta$ that maximizes the return $F(\theta)$.

### 1.2 What Exactly is $\theta$? (Flattening a Neural Network into a Vector)

In mathematical formulas, the neural network's parameters are written as a single vector $\theta \in \mathbb{R}^d$. To see what $\theta$ looks like in memory, consider a small 3-layer Multi-Layer Perceptron (MLP) with 5 input features, two hidden layers of 5 neurons each, and 1 output neuron:

* **Layer 1 ($5 \to 5$):** Weight matrix $W_1 \in \mathbb{R}^{5 \times 5}$ (25 weights) and bias vector $b_1 \in \mathbb{R}^5$ (5 biases).
* **Layer 2 ($5 \to 5$):** Weight matrix $W_2 \in \mathbb{R}^{5 \times 5}$ (25 weights) and bias vector $b_2 \in \mathbb{R}^5$ (5 biases).
* **Layer 3 ($5 \to 1$):** Weight matrix $W_3 \in \mathbb{R}^{5 \times 1}$ (5 weights) and bias scalar $b_3 \in \mathbb{R}^1$ (1 bias).

To treat the entire neural network as a single coordinate point in space, we unroll every matrix row by row and concatenate all weights and biases into a single 1D column vector $\theta$ of length $d = 25 + 5 + 25 + 5 + 5 + 1 = 66$:

```text
    [ Layer 1: W1 (5x5), b1 (5) ]       [ Layer 2: W2 (5x5), b2 (5) ]       [ Layer 3: W3 (5x1), b3 (1) ]
  ┌                               ┐   ┌                               ┐   ┌                               ┐
  │ w11  w12  w13  w14  w15 | b1  │   │ w'11 w'12 w'13 w'14 w'15| b'1 │   │      w''11  |  b''1           │
  │  :    :    :    :    :  |  :  │   │  :    :    :    :    :  |  :  │   │        :    |                 │
  │ w51  w52  w53  w54  w55 | b5  │   │ w'51 w'52 w'53 w'54 w'55| b'5 │   │      w''51  |                 │
  └                               ┘   └                               ┘   └                               ┘
                  │                                   │                                   │
                  └───────────────────────────────────┼───────────────────────────────────┘
                                                      ▼
                     Flattened 1D Parameter Vector (Genotype θ ∈ ℝ^66):
                     θ = [ w11, w12, ..., w55, b1, ..., b5, w'11, ..., b''1 ]ᵀ

```

**Key Takeaway on $\theta$:** Every element of $\theta$ is a floating-point number (`float32`). Adjusting $\theta$ changes the internal weights of the neural network, which changes how the agent reacts to states $s_t$.

## Module 2: Why Standard Deep Learning Breaks — Black Boxes and Non-Smoothness

In supervised deep learning, we optimize weights using **Backpropagation** (the Chain Rule of calculus). In reinforcement learning, direct backpropagation from the total return $F(\theta)$ back to the weights $\theta$ breaks down for two fundamental reasons.

### 2.1 Obstacle 1: The Environment is a "Black Box"

A system is a **White Box** when we know its exact algebraic equation and can differentiate through it. A neural network layer $a_t = \tanh(\theta \cdot s_t)$ is a white box because its derivative $\frac{\partial a_t}{\partial \theta}$ is known analytically.

By contrast, the environment is a **Black Box**: you can feed an action $a_t$ into the simulator and observe the resulting state $s_{t+1}$ and reward $r_t$, but you have no closed-form mathematical formula describing the internal C++ physics engine, collision detection, or Atari CPU memory registers.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. White Box (Neural Network): Explicit math formula is known           │
│                                                                         │
│    Input (s_t) ──► [ Formula: a_t = tanh(W · s_t + b) ] ──► Action (a_t)│
│                       (Derivative ∂a_t / ∂θ is known)                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Black Box (Atari / MuJoCo Simulator): Only Input & Output visible    │
│                                                                         │
│                   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                   │
│                   ┃   ？？？？？？？？？？？？？？？   ┃                   │
│   Action (a_t) ──►┃   Compiled Physics Engine,      ┃──► Return F(θ)    │
│                   ┃   Contact Solvers & if/else     ┃    & Next State   │
│                   ┃   Game Logic (No Formula!)      ┃    (s_{t+1})      │
│                   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛                   │
│               (Missing ∂s_{t+1}/∂a_t breaks the Chain Rule!)            │
└─────────────────────────────────────────────────────────────────────────┘

```

If we attempt to write the Chain Rule across an episode of length $T$:

$$\frac{\partial F(\theta)}{\partial \theta} = \underbrace{\frac{\partial R}{\partial s_T} \cdot \frac{\partial s_T}{\partial a_{T-1}}}_{\text{Inside Black Box (Unknown)}} \cdot \underbrace{\frac{\partial a_{T-1}}{\partial \theta}}_{\text{Inside Network (Known)}}$$

Because the environment transition derivatives $\frac{\partial s_{t+1}}{\partial a_t}$ are inaccessible, the multiplication chain is severed and standard backpropagation cannot be applied.

### 2.2 Obstacle 2: Continuous vs. Discrete Variables and Non-Smoothness

A common question is: *"If the total return $F(\theta)$ is a continuous floating-point number (like $45.25$), why don't its derivatives exist?"*

First, let us distinguish between **Continuous** and **Discrete** variables:

* **Continuous Variables:** Can take any real decimal value within an interval (e.g., motor torque $+0.731\text{ N}\cdot\text{m}$, joint angle, or total score $45.25$).
* **Discrete Variables:** Can only take distinct, separate options from a fixed set (e.g., pressing Atari joystick button `"LEFT"` vs. `"RIGHT"`, or selecting one of 10 quantized torque bins). You cannot press $0.42$ of the `"FIRE"` button.

Even when the final score $F(\theta)$ is a continuous real number, the **mapping** from weight $\theta$ to score $F(\theta)$ is often **non-smooth (piecewise constant or step-like)** whenever actions are discrete or physical collisions occur:

Suppose a single weight $\theta$ controls whether an agent moves `"LEFT"` (when $\theta < 2.0$, yielding score $F = 0$) or `"RIGHT"` (when $\theta \ge 2.0$, catching the ball and yielding score $F = +10$):

| Weight Value ($\theta$) | Chosen Action ($a$) | Episode Return $F(\theta)$ | Local Derivative ($\frac{dF}{d\theta}$) |
| --- | --- | --- | --- |
| $1.0000$ | Left | $0$ | $0$ (Flat plateau) |
| $1.9999$ | Left | $0$ | $0$ (Flat plateau) |
| $2.0000$ | Jump: Left $\to$ Right | Jump: $0 \to +10$ | $\infty$ (Undefined discontinuity) |
| $2.0001$ | Right | $+10$ | $0$ (Flat plateau) |
| $3.0000$ | Right | $+10$ | $0$ (Flat plateau) |

The table above shows why local calculus fails on step functions: everywhere except $\theta = 2.0$, the slope $\frac{dF}{d\theta}$ is exactly zero, giving the optimizer zero directional signal; right at $\theta = 2.0$, the function jumps vertically and the derivative does not exist.

```text
Episode Return F(θ)
   ▲
+10│                        ●━━━━━━━━━━━━━━━━━━━━━━━  (Upper plateau: slope = 0)
   │                        ┆
   │                        ┆ ◄── Vertical step at θ = 2.0:
   │                        ┆     Discontinuous jump (derivative undefined)
   │                        ┆
  0├━━━━━━━━━━━━━━━━━━━━━━━━○                         (Lower plateau: slope = 0)
   └───────────┴────────────┴────────────┴──────────► Weight (θ)
              1.0          2.0          3.0

```

### 2.3 The Solution: Gaussian Blur via Parameter Perturbation

Instead of evaluating a single point $\theta = 1.0$ and checking the slope directly beneath our feet, Evolution Strategies projects a **Gaussian probability cloud** of random test points $\theta + \sigma\epsilon$ around $\theta$:

```text
Episode Return
   ▲
+10│                        ●━━━━━━●━━━━━●━━━━━━━━━━  <-- Right-side samples (ε_3, ε_4)
   │                       ╱      ε_3   ε_4               land on the upper step (+10)!
   │                     ╱
   │                   ╱  ◄── Smoothed Objective E[F(θ+σε)] (Gaussian Blur):
   │                 ╱        The sharp cliff becomes a smooth, differentiable ramp!
  0├━━━━━●━━━━━★━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   └─────┴─────┴───┴────────┴───────────────────────► Weight (θ)
        ε_1  θ=1.0 ε_2     2.0

```

When centered at $\theta = 1.0$, most perturbed points land on the lower plateau ($F = 0$), but several points in the right tail of the cloud ($\epsilon_3, \epsilon_4$) cross $\theta = 2.0$ and score $+10$. Averaging across the cloud creates a smooth, differentiable curve $\mathbb{E}_\epsilon[F(\theta + \sigma\epsilon)]$ whose slope at $\theta = 1.0$ points directly toward the step!

## Module 3: Biological Intuition — Where is the "Evolution" in ES?

In the paper's equations, biological terms like *parent*, *offspring*, and *mutation* are encoded as linear algebra operations. Importantly, **there is only one single Parent network ($\theta_t$) per generation**.

### 3.1 Dictionary of Biological-to-Mathematical Terms

| Biological Concept | Math Symbol | Meaning in OpenAI ES |
| --- | --- | --- |
| **Genotype (DNA)** | $\theta \in \mathbb{R}^d$ | Flattened vector of all neural network weights. |
| **Generation** | $t \in \{0, 1, 2, \dots\}$ | One outer loop iteration of the algorithm. |
| **Single Parent** | $\theta_t$ | Master weight vector at the start of generation $t$. |
| **Mutation** | $+\sigma\epsilon_i$ | Random Gaussian noise vector added to the parent. |
| **Offspring ($i$-th child)** | $\tilde{\theta}_i = \theta_t + \sigma\epsilon_i$ | Perturbed neural network evaluated in one episode. |
| **Fitness** | $F_i = F(\theta_t + \sigma\epsilon_i)$ | Total scalar reward achieved by child $i$ in the game. |
| **Recombination** | $\theta_{t+1}$ | Weighted sum of all mutations that forms the next parent. |

### 3.2 Geometric Intuition: The "Tug-of-War" Recombination

Imagine the single parent $\theta_t$ standing at the center of a field. In generation $t$, the parent spawns $n$ offspring scattered in random directions $\epsilon_1, \dots, \epsilon_n$. Each child plays one full episode in the environment and pulls on the parent with a rope whose tension equals its fitness score $F_i$:

```text
                      Offspring 1 (θ_t + σε_1)
                      High Fitness: F_1 = +50
                         ▲  ▲  ▲
                         │  │  │ (Strong pull upward!)
                         │  │  │
Offspring 3 (θ_t + σε_3) ◄──┼──★──┼───► Offspring 4 (θ_t + σε_4)
Moderate: F_3 = +5       Parent θ_t     Zero Fitness: F_4 = 0 (No pull)
                            │
                            x (Pushes parent away due to negative score!)
                            ▼
                      Offspring 2 (θ_t + σε_2)
                      Poor Fitness: F_2 = -20

```

When we compute the update $\theta_{t+1} = \theta_t + \frac{\alpha}{n\sigma} \sum_{i=1}^n F_i \epsilon_i$:

* Offspring 1 scored $+50$ in direction $\epsilon_1$, pulling the parent strongly toward $\epsilon_1$.
* Offspring 2 scored $-20$ in direction $\epsilon_2$; multiplying $-20 \times \epsilon_2$ reverses the vector, pushing the parent away from $\epsilon_2$.
* The resulting vector sum becomes the **single new parent $\theta_{t+1}$** of the next generation.

## Module 4: Step-by-Step Mathematical Derivations

This section derives every equation in Section 2 of the paper from first principles.

### 4.1 Why CMA-ES Fails at Scale and Why We Use Isotropic Gaussians

Classical Evolution Strategies algorithms differ in how they represent the population cloud. The most famous variant, **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), models the population using a full-covariance multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$.

Compare the three ways to parameterize a Gaussian covariance matrix in $d$ dimensions:

```text
1. Full Covariance (CMA-ES)         2. Factored (Diagonal) Gaussian      3. Isotropic Gaussian (This Paper)
   Requires d × d parameters           Requires d independent parameters    Requires 1 fixed scalar (σ)!
   ┌                   ┐               ┌                   ┐                ┌                   ┐
   │ σ₁₁   σ₁₂   σ₁₃   │               │ σ₁²    0     0    │                │ σ²     0     0    │
   │ σ₂₁   σ₂₂   σ₂₃   │               │  0    σ₂²    0    │                │  0     σ²    0    │
   │ σ₃₁   σ₃₂   σ₃₃   │               │  0     0    σ₃²   │                │  0     0     σ²   │
   └                   ┘               └                   ┘                └                   ┘
  (Tilted ellipsoid in ℝ^d)           (Axis-aligned ellipsoid in ℝ^d)      (Perfectly symmetric sphere)

```

If a deep neural network has $d = 1,000,000$ weights, a full covariance matrix $\Sigma$ contains $d^2 = 10^{12}$ (one trillion) entries—requiring terabytes of memory and $O(d^3)$ matrix inversions. Consequently, CMA-ES is restricted to low-to-medium dimensions. By choosing an **Isotropic Multivariate Gaussian** with fixed covariance $\Sigma = \sigma^2 I$, OpenAI ES reduces covariance storage to a single scalar hyperparameter $\sigma$.

### 4.2 Dissecting the Probability Cloud $p_\psi(\theta)$ and the $(2\pi)^{d/2}$ Constant

In Natural Evolution Strategies (NES), the population is represented by a probability density function $p_\psi(\theta)$ governed by distribution parameters $\psi$:

* **$\theta \in \mathbb{R}^d$:** A sampled neural network parameter vector drawn from the cloud.
* **$\psi$:** The control knobs of the probability cloud itself (for a Gaussian, its center $\mu$ and spread $\sigma$).

For a $d$-dimensional isotropic Gaussian with mean $\mu \in \mathbb{R}^d$ and standard deviation $\sigma$, the probability density of drawing $\theta$ is:

$$p_\psi(\theta) = \frac{1}{(2\pi)^{d/2} \sigma^d} \exp\left( -\frac{\Vert{}\theta - \mu\Vert{}^2}{2\sigma^2} \right)$$

**Why is $(2\pi)^{d/2}$ in the denominator?**
For $p_\psi(\theta)$ to be a valid probability density function, the total volume (integral) under the bell curve across all $d$ dimensions must equal $1$. In 1D calculus, the Gaussian integral is $\int_{-\infty}^{\infty} e^{-x^2 / (2\sigma^2)} dx = \sqrt{2\pi}\sigma = (2\pi)^{1/2}\sigma$. When we multiply $d$ independent 1D Gaussians together to form a $d$-dimensional cloud, the normalization constant $\sqrt{2\pi}\sigma$ is multiplied by itself $d$ times:

$$\prod_{k=1}^{d} \frac{1}{\sqrt{2\pi}\sigma} = \frac{1}{\left(\sqrt{2\pi}\right)^d \sigma^d} = \frac{1}{(2\pi)^{d/2} \sigma^d}$$

### 4.3 Deriving Equation 1: The Score Function Estimator

NES seeks to maximize the expected objective value $\mathbb{E}_{\theta \sim p_\psi} [F(\theta)]$ over the population by ascending the gradient with respect to $\psi$:

$$\nabla_\psi \mathbb{E}_{\theta \sim p_\psi} F(\theta) = \mathbb{E}_{\theta \sim p_\psi} \Big\{ F(\theta) \, \nabla_\psi \log p_\psi(\theta) \Big\}$$

**Terminology Clarification — Fitness $F(\theta)$ vs. Score Function $\nabla_\psi \log p_\psi(\theta)$:**
Do not confuse the word "score" in game rewards with the statistical "Score Function":

* **$F(\theta)$ (Fitness / Return / Objective):** The actual reward earned by the agent in the environment.
* **$\nabla_\psi \log p_\psi(\theta)$ (The Statistical Score Function):** A classical term from mathematical statistics referring specifically to the gradient of the log-likelihood with respect to distribution parameters $\psi$. It acts as a **directional guide vector** pointing how to shift $\psi$ to make drawing sample $\theta$ more probable in the future.

**Step-by-Step Proof of Equation 1 (How Both Sides Equal Each Other):**

1. **Expand Expectation as an Integral:** By definition, the expectation of any function $g(\theta)$ under density $p_\psi(\theta)$ is $\mathbb{E}_{\theta \sim p_\psi}[g(\theta)] = \int g(\theta) p_\psi(\theta) d\theta$. Applying this to $F(\theta)$ gives:

$$\nabla_\psi \mathbb{E}_{\theta \sim p_\psi} F(\theta) = \nabla_\psi \int F(\theta) \, p_\psi(\theta) \, d\theta$$


2. **Move Gradient Inside the Integral:** Because $F(\theta)$ depends only on the weights $\theta$ and not on the distribution knobs $\psi$, the operator $\nabla_\psi$ passes cleanly through $F(\theta)$ and acts solely on $p_\psi(\theta)$:

$$= \int F(\theta) \, \nabla_\psi p_\psi(\theta) \, d\theta$$


3. **Apply the Log-Derivative Trick:** From basic calculus, the derivative of the natural logarithm of a function is $\nabla_\psi \log p_\psi(\theta) = \frac{\nabla_\psi p_\psi(\theta)}{p_\psi(\theta)}$. Multiplying both sides by $p_\psi(\theta)$ yields the identity:

$$\nabla_\psi p_\psi(\theta) = p_\psi(\theta) \, \nabla_\psi \log p_\psi(\theta)$$



Notice what this identity accomplishes: it takes a single $\nabla_\psi p_\psi(\theta)$ and splits it into **two** occurrences of $p_\psi(\theta)$—one isolated outside, and one trapped inside the $\log$ derivative!
4. **Substitute Back and Regroup as an Expectation:** Plugging the identity from Step 3 into the integral from Step 2 gives:

$$= \int F(\theta) \Big[ p_\psi(\theta) \, \nabla_\psi \log p_\psi(\theta) \Big] d\theta$$



Rearranging the multiplication order so that $p_\psi(\theta)$ sits next to $d\theta$:

$$= \int \underbrace{\Big( F(\theta) \, \nabla_\psi \log p_\psi(\theta) \Big)}_{g(\theta)} p_\psi(\theta) \, d\theta$$



By the exact definition of mathematical expectation from Step 1, any integral of the form $\int g(\theta) p_\psi(\theta) d\theta$ is written compactly as $\mathbb{E}_{\theta \sim p_\psi}[g(\theta)]$. Replacing the integral sign with $\mathbb{E}_{\theta \sim p_\psi}$ yields the right-hand side of Equation 1:

$$\nabla_\psi \mathbb{E}_{\theta \sim p_\psi} F(\theta) = \mathbb{E}_{\theta \sim p_\psi} \Big\{ F(\theta) \, \nabla_\psi \log p_\psi(\theta) \Big\}$$



### 4.4 Worked Numerical Example of the Score Function Update

Suppose we have a 1D neural network ($d = 1$) and our Gaussian cloud has parameters $\psi = (\mu, \sigma)$. Taking the logarithm of the 1D Gaussian density and differentiating with respect to $\mu$ and $\sigma$ gives:

$$\log p_\psi(\theta) = -\frac{1}{2}\ln(2\pi) - \ln(\sigma) - \frac{(\theta - \mu)^2}{2\sigma^2} \quad \Longrightarrow \quad \nabla_\psi \log p_\psi(\theta) = \begin{bmatrix} \frac{\partial \log p_\psi}{\partial \mu} \\ \frac{\partial \log p_\psi}{\partial \sigma} \end{bmatrix} = \begin{bmatrix} \frac{\theta - \mu}{\sigma^2} \\ \frac{(\theta - \mu)^2 - \sigma^2}{\sigma^3} \end{bmatrix}$$

Let $\mu_0 = 10$, $\sigma_0 = 2$, and learning rate $\alpha = 0.1$. We draw $n = 2$ samples:

* **Sample 1 ($\theta_1 = 8$, left of center):** Plays the game and gets a poor return $F(\theta_1) = -5$.

$$F(\theta_1) \cdot \frac{\theta_1 - \mu_0}{\sigma_0^2} = (-5) \times \frac{8 - 10}{2^2} = (-5) \times (-0.5) = +2.5$$



*(Because $\theta_1$ is to the left and got a negative score, the product of two negatives is $+2.5$—pushing the mean $\mu$ away from $8$ toward the right!)*
* **Sample 2 ($\theta_2 = 12$, right of center):** Plays the game and gets a high return $F(\theta_2) = +15$.

$$F(\theta_2) \cdot \frac{\theta_2 - \mu_0}{\sigma_0^2} = (+15) \times \frac{12 - 10}{2^2} = (+15) \times (+0.5) = +7.5$$



*(Pulls the mean $\mu$ strongly toward the right!)*
* **Combined Update:** Averaging the two gradients gives $\hat{\nabla}_\mu = \frac{2.5 + 7.5}{2} = +5.0$. Updating $\mu_1 = \mu_0 + \alpha \hat{\nabla}_\mu = 10 + 0.1(5.0) = 10.5$. The cloud center shifted toward the higher-scoring region without differentiating through the environment.

### 4.5 The Notation Shift & Deriving Equation 2 (Algorithm 1)

In the third paragraph of Section 2, the authors fix $\sigma$ as a constant hyperparameter and introduce an important **notation shift**:

* Instead of calling the mean of the Gaussian $\psi$ (or $\mu$) and calling the sampled network $\theta$, they now call the **mean vector itself $\theta$**!
* Any perturbed child parameter vector is written as $\tilde{\theta} = \theta + \sigma\epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$ is standard normal noise.

This reparameterizes the expected objective directly in terms of the parent parameters $\theta$:

$$\mathbb{E}_{\tilde{\theta} \sim p_\psi} F(\tilde{\theta}) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} F(\theta + \sigma\epsilon)$$

**Step-by-Step Proof that $\nabla_\theta \log p(\theta + \sigma\epsilon) = \frac{\epsilon}{\sigma}$:**

1. Write the log-density of $\tilde{\theta} \sim \mathcal{N}(\theta, \sigma^2 I)$:

$$\log p(\tilde{\theta}; \theta) = -\frac{d}{2}\ln(2\pi) - d\ln(\sigma) - \frac{\Vert{}\tilde{\theta} - \theta\Vert{}^2}{2\sigma^2}$$


2. Differentiate with respect to the mean vector $\theta$. The first two terms are constants with respect to $\theta$ and vanish:

$$\nabla_\theta \log p(\tilde{\theta}; \theta) = \nabla_\theta \left( -\frac{\Vert{}\tilde{\theta} - \theta\Vert{}^2}{2\sigma^2} \right) = -\frac{1}{2\sigma^2} \Big( -2(\tilde{\theta} - \theta) \Big) = \frac{\tilde{\theta} - \theta}{\sigma^2}$$


3. Substitute $\tilde{\theta} = \theta + \sigma\epsilon$ (so $\tilde{\theta} - \theta = \sigma\epsilon$) into the numerator:

$$\nabla_\theta \log p(\theta + \sigma\epsilon) = \frac{\sigma\epsilon}{\sigma^2} = \frac{\epsilon}{\sigma}$$



Plugging $\nabla_\theta \log p(\theta + \sigma\epsilon) = \frac{\epsilon}{\sigma}$ into the Score Function Estimator yields **Equation 2** of the paper:

$$\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} F(\theta + \sigma\epsilon) = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \Big\{ F(\theta + \sigma\epsilon) \, \epsilon \Big\}$$

Approximating the expectation $\mathbb{E}$ by averaging over $n$ sampled noise vectors $\epsilon_1, \dots, \epsilon_n \sim \mathcal{N}(0, I)$ gives the exact parameter update rule of **Algorithm 1**:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \frac{1}{n\sigma} \sum_{i=1}^{n} F_i \epsilon_i$$

## Module 5: Scaling to 1,440+ Cores — Algorithm 2 and Systems Engineering

A natural question arises: *"Doesn't evaluating a whole cloud of neural networks require vastly more computation than training a single network?"*

### 5.1 Why Probability Cloud Evaluation Does Not Explode Compute Time

While ES typically consumes **3x to 10x more environment frames** than policy gradient methods like A3C or TRPO, its wall-clock training time is often **10x to 100x faster** (e.g., solving 3D Humanoid in 10 minutes on 1,440 CPU cores versus ~1 day) due to four factors:

| Factor | Standard RL (A3C / TRPO) | Evolution Strategies (ES) | Practical Impact |
| --- | --- | --- | --- |
| **Per-Frame Compute** | Forward + Backprop + Value Net | Forward Pass Only | ~3x less compute per frame |
| **Network Bandwidth** | Sends full gradient ($\mathbb{R}^d$) | Sends 1 scalar return ($F_i$) | Near-zero communication bottleneck |
| **Value Estimation** | Sequential value catch-up | No value function | Scales across 1,000+ workers |
| **Noise Generation** | Sampled per step | Pre-allocated RAM noise table | O(1) array indexing overhead |

Because ES performs zero backpropagation and needs no value network, running 1 billion Atari frames in ES requires roughly the same total floating-point operations (FLOPs) as running 320 million frames in A3C.

### 5.2 Algorithm 2: The Shared Random Seed Trick

In standard distributed RL, if $n = 1,000$ workers compute gradients for a network with $d = 1,000,000$ parameters, every worker must transmit $4\text{ MB}$ of gradient floats across the network at every update, saturating network switches.

**Algorithm 2** removes this bottleneck using **Shared Random Seeds**:

1. Before training starts, all $n$ workers initialize an identical pseudo-random number generator (or pre-allocate an identical block of Gaussian noise in RAM) using known seeds.
2. During generation $t$, Worker $i$ samples noise $\epsilon_i$, runs one episode with parameters $\theta_t + \sigma\epsilon_i$, and obtains a single scalar number $F_i$.
3. Workers broadcast only their scalar return $F_i$ (4 bytes) and noise index to all other workers.
4. Every worker locally reconstructs $\epsilon_j$ for all other workers $j = 1, \dots, n$ from its own RAM table and executes the exact same update $\theta_{t+1} \leftarrow \theta_t + \frac{\alpha}{n\sigma}\sum_{j=1}^n F_j \epsilon_j$, staying synchronized without transmitting parameter vectors.

### 5.3 Six Engineering Stabilizers from Section 2.1

1. **Pre-Allocated Gaussian Noise Block:** Instead of calling a random number generator millions of times per generation, each worker allocates a large array of standard normal floats in RAM at startup. Sampling a perturbation vector $\epsilon_i \in \mathbb{R}^d$ simply slices a contiguous window `noise_table[idx : idx + d]` from a random starting index `idx`.
2. **Perturbing Parameter Subsets (Mixture of Gaussians):** For gigantic networks, workers can perturb only a subset of coordinates in $\theta$ while leaving others untouched. Mathematically, this corresponds to sampling from a **Mixture of Gaussians**, and in the extreme limit where each worker perturbs just a single coordinate, ES reduces to **Pure Finite Differences**.
3. **Antithetic (Mirrored) Sampling:** Workers always evaluate symmetric pairs of perturbations $+\epsilon$ and $-\epsilon$. Using a first-order Taylor expansion $F(\theta \pm \sigma\epsilon) \approx F(\theta) \pm \sigma \epsilon^T \nabla F(\theta)$, combining the pair cancels out the noisy baseline term $F(\theta)\epsilon$:

$$F(\theta + \sigma\epsilon)\epsilon + F(\theta - \sigma\epsilon)(-\epsilon) \approx \Big(F(\theta) + \sigma\epsilon^T\nabla F(\theta)\Big)\epsilon - \Big(F(\theta) - \sigma\epsilon^T\nabla F(\theta)\Big)\epsilon = 2\sigma (\epsilon^T\nabla F(\theta))\epsilon$$


4. **Fitness Shaping (Centered Rank Transformation):** Raw returns $F_i$ can have extreme outliers that destabilize gradient updates. Sorting the $n$ returns from lowest to highest and mapping their integer ranks $\{0, 1, \dots, n-1\}$ linearly onto $[-0.5, +0.5]$ normalizes the update magnitude across all environments.
5. **Weight Decay ($L_2$ Penalty) with Fixed $\sigma$:** Because the noise standard deviation $\sigma$ is held fixed throughout training, unchecked growth in the magnitude of weights $\Vert{}\theta\Vert{}$ would make fixed perturbations $\sigma\epsilon$ relatively tiny, killing exploration. Applying weight decay keeps $\Vert{}\theta\Vert{}$ proportional to $\sigma$.
6. **Dynamic Episode Length Capping ($m = 2\bar{L}$):** Because episodes vary in length, workers finishing early sit idle waiting for the longest episode. If $\bar{L}$ is the mean episode length across $n$ workers, total useful work is $n\bar{L}$. Capping every episode at $m = 2\bar{L}$ steps bounds total wall-clock CPU capacity by $n \cdot m = 2n\bar{L}$. Dividing useful work by maximum capacity proves worst-case CPU utilization remains at or above $50\%$:

$$\text{CPU Utilization} = \frac{\text{Useful Work}}{\text{Total Allocated Capacity}} \ge \frac{n \bar{L}}{n (2\bar{L})} = \frac{1}{2} = 50\%$$



## Module 6: Network Parameterization and Smoothing Space Comparison

### 6.1 Virtual Batch Normalization (VBN) for Atari

In Evolution Strategies, exploration comes entirely from parameter perturbations $\theta + \sigma\epsilon$. For ES to learn, different children must take different actions and earn different scores. In Atari games, adding Gaussian noise to standard Convolutional Neural Networks (CNNs) often produces degenerate policies that hold down a single button (e.g., always `"LEFT"`) regardless of what appears on screen, because random unnormalized layer biases dominate small pixel variations.

**Virtual Batch Normalization (VBN)** fixes this by selecting a fixed **reference minibatch** of game frames at the start of training. Whenever a perturbed network $\theta + \sigma\epsilon$ begins an episode, it passes the reference batch through its layers once to compute mean $\mu_{\text{ref}}$ and standard deviation $\sigma_{\text{ref}}$, and then normalizes every live game frame $z$ during the episode:

$$\hat{z} = \frac{z - \mu_{\text{ref}}}{\sqrt{\sigma_{\text{ref}}^2 + \delta}}$$

This centers the activations and amplifies tiny input pixel changes so random networks produce diverse actions. Because $\theta + \sigma\epsilon$ stays fixed for the entire episode (thousands of steps), $\mu_{\text{ref}}$ and $\sigma_{\text{ref}}$ are computed just once per episode, making the computational overhead negligible.

### 6.2 Action Discretization in MuJoCo

Conversely, in continuous control tasks like `Swimmer` and `Hopper`, a standard MLP with $\tanh$ activations can be *too smooth*: small parameter perturbations produce imperceptibly small changes in continuous motor torques, trapping the robot in uninformative local movements. Quantizing each continuous motor output into **10 uniform discrete bins** introduces deliberate step discontinuities that force small weight changes to trigger distinct, diverse locomotion gaits.

### 6.3 Parameter Space Smoothing (ES) vs. Action Space Smoothing (Policy Gradients)

Section 3 of the paper unifies Policy Gradients (PG) and Evolution Strategies (ES) as two ways of smoothing a non-differentiable decision problem $F(\theta) = R(a(\theta))$ over an episode of $T$ steps:

* **Policy Gradients (Action Space Smoothing):** Keeps $\theta$ fixed during the episode, but injects independent noise $\epsilon_t$ into the action choice at every single time step $t = 1, \dots, T$:

$$F_{\text{PG}}(\theta) = \mathbb{E}_\epsilon R(a(\epsilon, \theta)), \quad \nabla_\theta F_{\text{PG}}(\theta) = \mathbb{E}_\epsilon \Big\{ R(a(\epsilon, \theta)) \, \nabla_\theta \log p(a(\epsilon, \theta); \theta) \Big\}$$


* **Evolution Strategies (Parameter Space Smoothing):** Injects noise $\xi = \sigma\epsilon$ into the parameter vector **once** at the beginning of the episode ($\tilde{\theta} = \theta + \xi$) and runs the episode deterministically:

$$F_{\text{ES}}(\theta) = \mathbb{E}_\xi R(a(\xi, \theta)), \quad \nabla_\theta F_{\text{ES}}(\theta) = \mathbb{E}_\xi \Big\{ R(a(\xi, \theta)) \, \nabla_\theta \log p(\tilde{\theta}(\xi, \theta); \theta) \Big\}$$



**Why Parameter Space Smoothing Wins on Long Horizons ($T$):**
In Policy Gradients, because independent noise is injected at all $T$ steps, the variance of the gradient estimator grows linearly with episode length $O(T)$. To prevent variance explosion, PG algorithms must use reward discounting ($\gamma < 1$) and value function baselines, which makes them brittle when rewards are sparse or when frame-skipping changes the effective horizon $T$. In ES, noise is sampled only **once per episode** in parameter space, making gradient variance largely independent of episode length $T$ and removing the need for discounting or value functions.
