# 🧠 Differential Calculus Cheatsheet for AI/ML (Extended)

> Mastering derivatives is foundational for understanding gradients, loss optimization, and backpropagation in modern ML models.

---

## ✏️ 1. Basic Derivatives Recap

### Definition:
$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

- Measures **instantaneous rate of change** (slope)
- Fundamental to **gradient-based optimization**

---

## 🔄 2. Derivative Rules (Quick Reference)

| Rule Type       | Formula | Description |
|----------------|---------|-------------|
| Power Rule     | \( \frac{d}{dx}[x^n] = nx^{n-1} \) | For polynomial terms |
| Chain Rule     | \( \frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x) \) | Backpropagation core |
| Product Rule   | \( \frac{d}{dx}[fg] = f'g + fg' \) | Needed for loss * activation |
| Quotient Rule  | \( \frac{d}{dx}\left[\frac{f}{g}\right] = \frac{f'g - fg'}{g^2} \) | Used with division terms |
| Exponential    | \( \frac{d}{dx}[e^{f(x)}] = e^{f(x)} \cdot f'(x) \) | Appears in softmax, regularization |
| Logarithmic    | \( \frac{d}{dx}[\ln(f(x))] = \frac{f'(x)}{f(x)} \) | Key in log-likelihood loss |

---


## 🔣 3. Derivatives of Key ML Functions

| Function         | Derivative                              | Application                      |
|------------------|------------------------------------------|----------------------------------|
| \( \sigma(x) = \frac{1}{1 + e^{-x}} \) | \( \sigma(x)(1 - \sigma(x)) \)       | Sigmoid activation               |
| \( \tanh(x) \)    | \( 1 - \tanh^2(x) \)                     | RNNs/Activations                 |
| \( \text{ReLU}(x) \) | \( 1 \text{ if } x > 0, \ 0 \text{ otherwise} \) | CNNs/Activations          |
| \( \log(x) \)     | \( \frac{1}{x} \)                        | Cross-entropy, likelihood        |
| \( e^x \)         | \( e^x \)                                | Softmax, regularization          |

---

## 🧮 4. Partial Derivatives (Multivariable Calculus)

For a function \( f(x, y) \), the **partial derivative** with respect to \( x \) is:
$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
$$

Used when functions depend on multiple variables (weights, inputs).

**Example**:  
Given a loss function \( J(w, b) \), compute:
- \( \frac{\partial J}{\partial w} \) (gradient w.r.t weights)
- \( \frac{\partial J}{\partial b} \) (gradient w.r.t bias)

---

## 🧭 5. Gradient Vector (∇f)

For a scalar-valued function \( f: \mathbb{R}^n \to \mathbb{R} \), the **gradient** is:
$$
\nabla f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]^T
$$

- Points in the direction of **steepest ascent**
- Used in **gradient descent**:  
  $$
  \theta := \theta - \alpha \cdot \nabla J(\theta)
  $$

---

## 🧾 6. Jacobian Matrix

For a vector-valued function \( \mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m \), the **Jacobian** is:
$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

- Shape: \( m \times n \)
- Used in **vector-valued backpropagation**, e.g. in deep learning frameworks.

---

## 🧠 7. Hessian Matrix

Second-order partial derivatives:
$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

- Matrix of curvature
- Used in **Newton’s Method**, convex optimization
- Costly in high dimensions, but useful for understanding local behavior

---

## 💡 8. Useful Derivatives in ML

| Expression                           | Derivative                                         |
|--------------------------------------|----------------------------------------------------|
| \( \mathbf{w}^T \mathbf{x} \)         | \( \frac{d}{d\mathbf{w}} = \mathbf{x} \)           |
| \( ||\mathbf{w}||^2 \)               | \( \frac{d}{d\mathbf{w}} = 2\mathbf{w} \)          |
| \( \log(\sigma(x)) \)                | \( \frac{1}{\sigma(x)} \cdot \sigma(x)(1 - \sigma(x)) = 1 - \sigma(x) \) |
| Softmax: \( \sigma_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \) |  
\( \frac{\partial \sigma_i}{\partial z_k} = \sigma_i (\delta_{ik} - \sigma_k) \) | Cross-entropy + softmax combo

---

## 🔁 9. Backpropagation Essentials

- Derivatives are propagated backward using **chain rule**:
  $$
  \frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
  $$

- Applies layer by layer in neural networks:
  - Forward pass: compute activations
  - Backward pass: compute gradients using chain rule

---

## 🛠 10. Tips for Working with Derivatives in ML

- Use **symbolic differentiation** (e.g., SymPy) for derivation
- Use **automatic differentiation** (e.g., PyTorch, TensorFlow) in practice
- Understand how to derive manually for interpretability and debugging

---

## 📌 Summary

| Concept             | Symbol / Tool           | Usage in ML                           |
|---------------------|--------------------------|----------------------------------------|
| Derivative          | \( \frac{df}{dx} \)       | 1D gradients                          |
| Partial Derivative  | \( \frac{\partial f}{\partial x_i} \) | Multivariable models             |
| Gradient            | \( \nabla f \)            | Optimization direction                |
| Jacobian            | \( \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \) | Vector-valued loss functions |
| Hessian             | \( \nabla^2 f \)          | Curvature for 2nd-order methods       |

---


