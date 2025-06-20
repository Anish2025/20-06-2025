# 🧠 Differential Calculus Cheatsheet for AI/ML

> **Why It Matters in AI/ML**: Differential calculus powers optimization—used in gradient descent, backpropagation, cost minimization, etc.

---

## ✏️ Basics of Derivatives

**Definition**  
The derivative of a function \( f(x) \) is:  
\[
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\]

**Interpretation**:  
- Measures rate of change (slope)  
- Tangent to the curve at point \( x \)

---

## ⚙️ Common Derivative Rules

| Rule Name         | Formula                                      | Example                            |
|------------------|----------------------------------------------|------------------------------------|
| Power Rule        | \( \frac{d}{dx}[x^n] = nx^{n-1} \)            | \( \frac{d}{dx}[x^3] = 3x^2 \)     |
| Constant Rule     | \( \frac{d}{dx}[c] = 0 \)                     | \( \frac{d}{dx}[5] = 0 \)          |
| Constant Mult.    | \( \frac{d}{dx}[cf(x)] = c f'(x) \)           | \( \frac{d}{dx}[3x^2] = 6x \)      |
| Sum Rule          | \( \frac{d}{dx}[f + g] = f' + g' \)           | \( \frac{d}{dx}[x^2 + e^x] = 2x + e^x \) |
| Product Rule      | \( \frac{d}{dx}[fg] = f'g + fg' \)            | \( \frac{d}{dx}[x \cdot \ln x] \)  |
| Quotient Rule     | \( \frac{d}{dx}\left[\frac{f}{g}\right] = \frac{f'g - fg'}{g^2} \) | \( \frac{d}{dx}\left[\frac{1}{x}\right] = -\frac{1}{x^2} \) |
| Chain Rule        | \( \frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x) \) | \( \frac{d}{dx}[\sin(x^2)] = \cos(x^2) \cdot 2x \) |

---

## 🔣 Derivatives of Common Functions

| Function       | Derivative         |
|----------------|--------------------|
| \( x^n \)       | \( nx^{n-1} \)      |
| \( \ln x \)     | \( \frac{1}{x} \)   |
| \( e^x \)       | \( e^x \)           |
| \( \sin x \)    | \( \cos x \)        |
| \( \cos x \)    | \( -\sin x \)       |
| \( \tan x \)    | \( \sec^2 x \)      |

---

## 📈 Higher Order Derivatives

- 2nd Derivative: \( f''(x) = \frac{d^2f}{dx^2} \)
- Use: Concavity, acceleration, local min/max detection

---

## 🧮 Applications in AI/ML

| Concept                | Use                                                |
|------------------------|-----------------------------------------------------|
| Gradient Descent       | Uses partial derivatives to update weights          |
| Cost Function Optimization | Find minimum using \( \nabla J(\theta) \)    |
| Backpropagation        | Applies chain rule through layers                   |
| Jacobian Matrix        | Partial derivatives of vector-valued functions      |
| Hessian Matrix         | Matrix of second-order derivatives (curvature info) |

---

## 🔍 Tips & Tricks

- Use tools like **SymPy**, **NumPy**, or **WolframAlpha** for symbolic derivation
- Always simplify functions before differentiating
- For multivariable calculus, learn **partial derivatives**

---

## 📘 Example: Gradient Descent

\[
\theta := \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta}
\]

Where:
- \( J(\theta) \): Loss function  
- \( \alpha \): Learning rate  
- \( \frac{\partial J}{\partial \theta} \): Gradient of loss wrt parameters  

---
