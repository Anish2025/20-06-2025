# Differential Calculus Cheatsheet for AI/ML

> Mastering derivatives is foundational for understanding gradients, loss optimization, and backpropagation in modern ML models.

---

## 1. Basic Derivative Definition   

The derivative of a function \( f(x) \) measures how it changes with respect to \( x \). It is defined as:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

- Describes the **rate of change**
- Used to compute slopes and sensitivity in models
- Core to **gradient descent** and **loss minimization**

[Basic Derivative Explanation](https://youtu.be/7K1sB05pE0A?si=z4_Lp8N0UmD-1uOT)

---

## 2. Fundamental Derivative Rules

### Power Rule

$$
\frac{d}{dx}[x^n] = nx^{n-1}
$$  


Used for polynomial terms and cost functions.

### Chain Rule

$$
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
$$  

Essential for backpropagation in neural networks.

### Product Rule

$$
\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)
$$  
 
Used when two functions (like input × weight) are multiplied.

### Quotient Rule

$$
\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
$$  

Helpful when dividing functions in normalized outputs.

### Exponential Rule

$$
\frac{d}{dx}[e^{f(x)}] = e^{f(x)} \cdot f'(x)
$$  

Used in softmax, attention mechanisms, and regularization terms.

### Logarithmic Rule

$$
\frac{d}{dx}[\ln(f(x))] = \frac{f'(x)}{f(x)}
$$  

Common in log-likelihood and entropy-based loss functions.

[Chain, Product, Quotient Rule and Higher Degree Derivatives Explanation](https://www.youtube.com/watch?v=4sTKcvYMNxk&t=165s)

---

## 3. Derivatives of Key ML Functions

### Sigmoid Activation [Explanation](https://youtu.be/TPqr8t919YM?si=dBlESpGjOuIGyfjo)
$$
\sigma(x) = \frac{1}{1 + e^{-x}} \\
\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
$$

### Tanh Activation
$$
\frac{d}{dx}[\tanh(x)] = 1 - \tanh^2(x)
$$

### ReLU Activation
$$
\frac{d}{dx}[\text{ReLU}(x)] = 
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases}
$$

### Logarithm
$$
\frac{d}{dx}[\log(x)] = \frac{1}{x}
$$

### Exponential
$$
\frac{d}{dx}[e^x] = e^x
$$

[Sigmoid, TenH, ReLU Explanation](https://www.youtube.com/watch?v=QBHj9xoPy_o)

---

## 4. Partial Derivatives

Used when functions have multiple inputs.

For a function \( f(x, y) \), the partial derivative with respect to \( x \) is:

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
$$

In ML, used to compute how the loss changes with respect to each parameter (weight or bias).

**Example**:  
If is a loss function, then:

$$
\( J(w, b) \)
$$

the rate of change of the loss with respect to the weight \( w \)

$$
\( \frac{\partial J}{\partial w} \) 
$$

the rate of change of the loss with respect to the bias \( b \)


$$
 \( \frac{\partial J}{\partial b} \)
$$

[Partial Derivatives Explanation](https://youtu.be/KnVNFj53Eq4?si=9Vv9P-78SDnl_0i_)

---

## 5. Gradient Vector ( ∇f )

The gradient is a vector of partial derivatives for multivariable functions.

If 

$$
\( f: \mathbb{R}^n \rightarrow \mathbb{R} \)
$$

, then:

$$
\nabla f(\mathbf{x}) = 
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

In ML, the gradient tells the direction to update model parameters to **reduce loss**.

### Gradient Descent Update Rule:

$$
\theta := \theta - \alpha \cdot \nabla J(\theta)
$$

Where:

$$
\( \theta \):  model parameter vector
$$

$$
 \( \alpha \): learning rate
$$

$$
\( J(\theta) \): loss function
$$

[Gradient Vector Explanation](https://youtu.be/yXD5IlDstNk?si=fMG-JYLjLbPWBEmX)

---

## 6. Jacobian Matrix

The Jacobian generalizes the gradient to vector-valued functions.

For 

$$
\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

Then the Jacobian matrix is defined as:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

Used in:
- Multivariate output models
- Neural network layer transformations
- Vectorized backpropagation
  
[Jacobian Matrix Explanation](https://youtu.be/bohL918kXQk?si=Fi70lh06SntgRLsx)

---

## 7. Hessian Matrix

Second-order derivatives that represent curvature of a function.

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

- Square matrix: \( n \times n \)
- Helps in **Newton’s Method**, convexity checks
- Not commonly computed in deep learning due to high cost

[Hessian Matrix Explanation](https://youtu.be/LbBcuZukCAw?si=F0N-NVd04mINABA6)

---

## 8. ML-Specific Derivative Expressions

### Dot Product Derivative

If 

$$
\( f(\mathbf{w}) = \mathbf{w}^T \mathbf{x} \)
$$

, then:

$$
\frac{d}{d\mathbf{w}} = \mathbf{x}
$$

### L2 Norm (Weight Regularization)

$$
f(\mathbf{w}) = ||\mathbf{w}||^2 = \mathbf{w}^T \mathbf{w} \\
\frac{d}{d\mathbf{w}} = 2\mathbf{w}
$$

### Log of Sigmoid

$$
\frac{d}{dx}[\log(\sigma(x))] = 1 - \sigma(x)
$$

### Softmax Derivative

Let:

$$
\sigma_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Then:

$$
\frac{\partial \sigma_i}{\partial z_k} = \sigma_i (\delta_{ik} - \sigma_k)
$$

Used in:
- Softmax + cross-entropy derivation
- Classification gradients

[L2 Norm Explanation & Proof](https://youtu.be/uu2X47cSLmM?si=CbNtkgO16gekCOyj)

[Softmax Derivative Explanation & Proof](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)

---

## 9. Backpropagation

Backpropagation relies heavily on the **chain rule**.

If:

$$
L = f(g(x)) \quad \Rightarrow \quad \frac{dL}{dx} = \frac{dL}{dg} \cdot \frac{dg}{dx}
$$

In neural networks:
- **Forward pass** computes activations
- **Backward pass** computes gradients layer-by-layer using chain rule

[Backpropagation Explanation](https://youtu.be/tIeHLnjs5U8?si=qRgpIG6dDkPfhSlt)

[Proof](https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf)

---

## 10. Practical Tips

- Use symbolic tools like **SymPy** to derive expressions.
- Use **automatic differentiation** libraries (PyTorch, TensorFlow).
- Understand derivatives deeply for **debugging training issues** and **custom layer implementation**.

---

## Summary

- **Derivative** → Instant rate of change
- **Partial Derivative** → Change wrt one input in multivariable function
- **Gradient** → Vector of partials (used in gradient descent)
- **Jacobian** → Matrix of partials (vector-valued functions)
- **Hessian** → Matrix of second-order partials (used in optimization)

---
## Final Formula Summary

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

### Common Function Derivatives

| Function             | Derivative                                                                         |
|----------------------|------------------------------------------------------------------------------------|
| eˣ                  | d/dx[eˣ] = eˣ                                                                       |
| log(x)              | d/dx[log(x)] = 1/x                                                                 |
| tanh(x)             | d/dx[tanh(x)] = 1 - tanh²(x)                                                        |
| sigmoid(x)          | d/dx[σ(x)] = σ(x)·(1 - σ(x))                                                        |
| ReLU(x)             | d/dx[ReLU(x)] = 1 if x > 0, else 0                                                  |

---

### Vector & Matrix Derivatives

| Expression                          | Result                                                               |
|-------------------------------------|----------------------------------------------------------------------|
| ∇f                                 | [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ                                       |
| Jacobian                           | Jᵢⱼ = ∂fᵢ / ∂xⱼ                                                      |
| Hessian                            | Hᵢⱼ = ∂²f / ∂xᵢ∂xⱼ                                                  |

---

### ML-Specific Derivatives

| Expression                          | Derivative                                                           |
|-------------------------------------|----------------------------------------------------------------------|
| wᵀ·x                                | d/dw = x                                                             |
| ||w||²                              | d/dw = 2w                                                            |
| log(σ(x))                           | d/dx = 1 - σ(x)                                                      |
| softmax(σᵢ)                         | ∂σᵢ/∂zₖ = σᵢ(δᵢₖ - σₖ)                                               |


1. **3Blue1Brown – Essence of Calculus (Video Series)**  
   [https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)  
   → Intuitive and visual understanding of differentiation and gradients.
2. **Wikipedia – Calculus & Differential Calculus**  
   [https://en.wikipedia.org/wiki/Differential_calculus](https://en.wikipedia.org/wiki/Differential_calculus)  
   → General reference for notations and extended rules.
   3.**MIT OpenCourseWare – Single Variable Calculus**  
   [https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010)  
   → Foundation of derivatives, rules, and real-world interpretation.



