"""
A minimal reverse-mode autograd engine over NumPy arrays.

Why does an alignment/interpretability repo ship its own autograd engine
instead of depending on PyTorch?

  1. This repo's sandbox has no route to huggingface.co (only PyPI/GitHub/npm
     mirrors are reachable), so real pretrained weights can't be downloaded
     here regardless of framework. The honest move is to train a small
     reference model from scratch and be explicit about that (see
     docs/plan_a_research.md and STATUS.md).
  2. A ~200 line autograd engine is something a reader can hold in their head
     completely -- every gradient rule is visible. For a tool whose entire
     purpose is "look inside the black box," it felt wrong to make the
     training engine itself a black box.
  3. It avoids an unnecessary ~1GB CUDA toolkit dependency for a model with
     4,000 parameters.

This is intentionally NOT a general-purpose ML framework. It implements just
the ops the toy transformer, SAE, and probes in this package need. If you
plug in real HuggingFace models via neuralese/hf_adapter.py, you'll use
PyTorch there instead -- this engine is only for the from-scratch reference
model.
"""
from __future__ import annotations
import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing whatever NumPy broadcasting did."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """A NumPy array with a backward() method. See module docstring."""

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- bookkeeping -----------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def item(self):
        return float(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # -- elementwise ops ---------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def reciprocal(self):
        out = Tensor(1.0 / self.data, self.requires_grad, (self,), "recip")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (-1.0 / (self.data ** 2))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.reciprocal()

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(g, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(g, other.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), self.requires_grad, (self,), "sqrt")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (0.5 / out.data)

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    # -- reductions / reshaping -------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = 1
            for a in axes:
                n *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def swapaxes(self, a, b):
        out = Tensor(np.swapaxes(self.data, a, b), self.requires_grad, (self,), "swapaxes")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.swapaxes(out.grad, a, b)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes), self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    self.grad += np.transpose(out.grad, np.argsort(axes))

        out._backward = _backward
        return out

    def __getitem__(self, index):
        out = Tensor(self.data[index], self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, index, out.grad)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                dy = out.grad
                dot = np.sum(dy * s, axis=axis, keepdims=True)
                self.grad += s * (dy - dot)

        out._backward = _backward
        return out

    # -- graph traversal ---------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self._ensure_grad()
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def embedding_lookup(table: Tensor, indices: np.ndarray) -> Tensor:
    """Row-gather with a scatter-add backward. indices: int array, any shape."""
    out = Tensor(table.data[indices], table.requires_grad, (table,), "embed")

    def _backward():
        if table.requires_grad:
            table._ensure_grad()
            np.add.at(table.grad, indices, out.grad)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray):
    """logits: (N, vocab) Tensor. targets: (N,) int array. Returns (loss_tensor, probs)."""
    z = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    e = np.exp(z)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    n = logits.data.shape[0]
    nll = -np.log(probs[np.arange(n), targets] + 1e-12)
    out = Tensor(np.mean(nll), logits.requires_grad, (logits,), "xent")

    def _backward():
        if logits.requires_grad:
            logits._ensure_grad()
            grad = probs.copy()
            grad[np.arange(n), targets] -= 1
            grad /= n
            logits.grad += grad * out.grad

    out._backward = _backward
    return out, probs


class Adam:
    """Standard Adam optimizer operating directly on Tensor parameters."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
