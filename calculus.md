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

---

## 🔣 3. Derivatives of Key ML Functions

### Sigmoid Activation
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

---

## 🧮 4. Partial Derivatives

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



---

## 🧭 5. Gradient Vector ( ∇f )

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
- \( \theta \): model parameter vector
- \( \alpha \): learning rate
- \( J(\theta) \): loss function

---

## 🧾 6. Jacobian Matrix

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

---

## 🧠 7. Hessian Matrix

Second-order derivatives that represent curvature of a function.

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

- Square matrix: \( n \times n \)
- Helps in **Newton’s Method**, convexity checks
- Not commonly computed in deep learning due to high cost

---

## 💡 8. ML-Specific Derivative Expressions

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

---

## 🔁 9. Backpropagation

Backpropagation relies heavily on the **chain rule**.

If:

$$
L = f(g(x)) \quad \Rightarrow \quad \frac{dL}{dx} = \frac{dL}{dg} \cdot \frac{dg}{dx}
$$

In neural networks:
- **Forward pass** computes activations
- **Backward pass** computes gradients layer-by-layer using chain rule

---

## 🛠 10. Practical Tips

- Use symbolic tools like **SymPy** to derive expressions.
- Use **automatic differentiation** libraries (PyTorch, TensorFlow).
- Understand derivatives deeply for **debugging training issues** and **custom layer implementation**.

---

## 📌 Summary

- **Derivative** → Instant rate of change
- **Partial Derivative** → Change wrt one input in multivariable function
- **Gradient** → Vector of partials (used in gradient descent)
- **Jacobian** → Matrix of partials (vector-valued functions)
- **Hessian** → Matrix of second-order partials (used in optimization)

---

