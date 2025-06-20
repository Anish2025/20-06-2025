## 📘 Final Formula Summary

| Concept/Rule         | Formula (Plaintext)                                                                 |
|----------------------|--------------------------------------------------------------------------------------|
| Derivative Definition| f'(x) = lim(h→0) [f(x+h) - f(x)] / h                                                 |
| Power Rule           | d/dx[xⁿ] = n·xⁿ⁻¹                                                                   |
| Chain Rule           | d/dx[f(g(x))] = f'(g(x)) · g'(x)                                                    |
| Product Rule         | d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x)                                            |
| Quotient Rule        | d/dx[f(x)/g(x)] = [f'(x)·g(x) - f(x)·g'(x)] / g(x)²                                 |
| Exponential Rule     | d/dx[e^{f(x)}] = e^{f(x)} · f'(x)                                                   |
| Logarithmic Rule     | d/dx[ln(f(x))] = f'(x) / f(x)                                                       |

---

### 🔣 Common Function Derivatives

| Function             | Derivative                                                                         |
|----------------------|------------------------------------------------------------------------------------|
| eˣ                  | d/dx[eˣ] = eˣ                                                                       |
| log(x)              | d/dx[log(x)] = 1/x                                                                 |
| tanh(x)             | d/dx[tanh(x)] = 1 - tanh²(x)                                                        |
| sigmoid(x)          | d/dx[σ(x)] = σ(x)·(1 - σ(x))                                                        |
| ReLU(x)             | d/dx[ReLU(x)] = 1 if x > 0, else 0                                                  |

---

### 🧮 Vector & Matrix Derivatives

| Expression                          | Result                                                               |
|-------------------------------------|----------------------------------------------------------------------|
| ∇f                                 | [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ                                       |
| Jacobian                           | Jᵢⱼ = ∂fᵢ / ∂xⱼ                                                      |
| Hessian                            | Hᵢⱼ = ∂²f / ∂xᵢ∂xⱼ                                                  |

---

### 🤖 ML-Specific Derivatives

| Expression                          | Derivative                                                           |
|-------------------------------------|----------------------------------------------------------------------|
| wᵀ·x                                | d/dw = x                                                             |
| ||w||²                              | d/dw = 2w                                                            |
| log(σ(x))                           | d/dx = 1 - σ(x)                                                      |
| softmax(σᵢ)                         | ∂σᵢ/∂zₖ = σᵢ(δᵢₖ - σₖ)                                               |
