# 🧠 Differential Calculus Cheatsheet for AI/ML (Extended)

> Mastering derivatives is foundational for understanding gradients, loss optimization, and backpropagation in modern ML models.

---

## ✏️ 1. Basic Derivative Definition

The derivative of a function \( f(x) \) measures how it changes with respect to \( x \). It is defined as:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

- Describes the **rate of change**
- Used to compute slopes and sensitivity in models
- Core to **gradient descent** and **loss minimization**

---

## 🔄 2. Fundamental Derivative Rules

### Power Rule

$$
\frac{d}{dx}[x^n] = nx^{n-1}
$$  

### Chain Rule

$$
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
$$  

### Product Rule

$$
\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)
$$  

### Quotient Rule

$$
\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
$$  

### Exponential Rule

$$
\frac{d}{dx}[e^{f(x)}] = e^{f(x)} \cdot f'(x)
$$  

### Logarithmic Rule

$$
\frac{d}{dx}[\ln(f(x))] = \frac{f'(x)}{f(x)}
$$  

---

## 🔣 3. Derivatives of Key ML Functions

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \\
\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
$$

### Tanh

$$
\frac{d}{dx}[\tanh(x)] = 1 - \tanh^2(x)
$$

### ReLU

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

---

## 🧮 4. Partial Derivatives

For multivariable functions like \( f(x, y) \):

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
$$

---

## 🧭 5. Gradient Vector ( ∇f )

For scalar-valued multivariable functions:

$$
\nabla f(\mathbf{x}) = 
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

### Gradient Descent:

$$
\theta := \theta - \alpha \cdot \nabla J(\theta)
$$

---

## 🧾 6. Jacobian Matrix

For \( \mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m \):

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

---

## 🧠 7. Hessian Matrix

Second-order derivatives:

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

---

## 💡 8. ML-Specific Derivative Expressions

### Dot Product

$$
\frac{d}{d\mathbf{w}}(\mathbf{w}^T \mathbf{x}) = \mathbf{x}
$$

### L2 Norm

$$
\frac{d}{d\mathbf{w}}(||\mathbf{w}||^2) = 2\mathbf{w}
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

---

## 🔁 9. Backpropagation Chain Rule

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
$$

---

## 🛠 10. Practical Tips

- Use symbolic tools like **SymPy** to derive expressions.
- Use **autodiff** tools like PyTorch or TensorFlow.
- Know when to simplify by hand for debugging.

---

## 📘 Final Formula Summary

Below is a quick collection of all formulas mentioned:

### Derivative Definition

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

### Power Rule

$$
\frac{d}{dx}[x^n] = nx^{n-1}
$$  

### Chain Rule

$$
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
$$  

### Product Rule

$$
\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)
$$  

### Quotient Rule

$$
\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
$$  

### Exponential Rule

$$
\frac{d}{dx}[e^{f(x)}] = e^{f(x)} \cdot f'(x)
$$  

### Logarithmic Rule

$$
\frac{d}{dx}[\ln(f(x))] = \frac{f'(x)}{f(x)}
$$  

### Common Function Derivatives

-

   $$
  \( \frac{d}{dx}[e^x] = e^x \)
  $$

  -

   $$
 \( \frac{d}{dx}[\log(x)] = \frac{1}{x} \)
  $$

  -

   $$
  \( \frac{d}{dx}[\tanh(x)] = 1 - \tanh^2(x) \)
  $$

  -

   $$
  \( \frac{d}{dx}[\sigma(x)] = \sigma(x)(1 - \sigma(x)) \)
  $$

  -

   $$
  \( \frac{d}{dx}[\text{ReLU}(x)] = 1 \text{ if } x > 0; 0 \text{ otherwise} \)
  $$
  
### Vectors & Matrices
 -

   $$
 \( \nabla f = \left[ \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right]^T \)
  $$

 -

   $$
  \( J_{ij} = \frac{\partial f_i}{\partial x_j} \)
  $$

   -

   $$
  \( H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} \)
  $$


### ML-specific

 -

   $$
 \( \frac{d}{d\mathbf{w}}(\mathbf{w}^T \mathbf{x}) = \mathbf{x} \)
  $$

   -

   $$
   \( \frac{d}{d\mathbf{w}}(||\mathbf{w}||^2) = 2\mathbf{w} \)
  $$

   -

   $$
  \( \frac{d}{dx}[\log(\sigma(x))] = 1 - \sigma(x) \)
  $$

   -

   $$
  \( \frac{\partial \sigma_i}{\partial z_k} = \sigma_i (\delta_{ik} - \sigma_k) \)
  $$
  

---


