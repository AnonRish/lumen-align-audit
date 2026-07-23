"""Gradient-check every autograd primitive against finite differences.
This is the foundation everything else in neuralese/ is built on, so it gets
checked in isolation before anything downstream trusts it."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.neuralese.tensor import Tensor, embedding_lookup, cross_entropy, Adam

TOL = 1e-4


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f1 = f()
        x[idx] = orig - eps
        f2 = f()
        x[idx] = orig
        grad[idx] = (f1 - f2) / (2 * eps)
    return grad


def assert_gradient_correct(build_fn, x_arrays, tol=TOL):
    tensors = [Tensor(x.copy(), requires_grad=True) for x in x_arrays]
    loss = build_fn(tensors)
    loss.backward()
    analytic = [t.grad.copy() for t in tensors]
    for i, x in enumerate(x_arrays):
        def f(i=i):
            ts = [Tensor(xa.copy()) for xa in x_arrays]
            return build_fn(ts).data
        num = numerical_grad(f, x)
        err = np.max(np.abs(num - analytic[i]))
        assert err < tol, f"operand {i}: max_abs_err={err:.2e}"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_add_mul_broadcast(rng):
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((4,))
    assert_gradient_correct(lambda ts: (ts[0] + ts[1] * ts[0]).sum(), [a, b])


def test_matmul_batched_broadcast(rng):
    x, w = rng.standard_normal((2, 5, 6)), rng.standard_normal((6, 3))
    assert_gradient_correct(lambda ts: (ts[0] @ ts[1]).sum(), [x, w])


def test_matmul_batched_equal_shapes(rng):
    """The exact shape pattern attention uses: (B,H,S,Dh) @ (B,H,Dh,S), no broadcasting."""
    q, k = rng.standard_normal((2, 3, 4, 5)), rng.standard_normal((2, 3, 4, 5))
    assert_gradient_correct(lambda ts: (ts[0] @ ts[1].swapaxes(-1, -2)).sum(), [q, k])


def test_relu_sqrt_chain(rng):
    c = rng.standard_normal(5)
    assert_gradient_correct(lambda ts: ((ts[0] * ts[0] + 1.0).sqrt()).relu().sum(), [c])


def test_softmax(rng):
    d = rng.standard_normal((3, 5))
    assert_gradient_correct(lambda ts: (ts[0].softmax(axis=-1) * ts[0].softmax(axis=-1)).sum(), [d])


def test_mean(rng):
    e = rng.standard_normal((4, 7))
    assert_gradient_correct(lambda ts: ts[0].mean(axis=1).sum(), [e])


def test_swapaxes(rng):
    f = rng.standard_normal((2, 3, 4))
    assert_gradient_correct(lambda ts: ts[0].swapaxes(1, 2).sum() * 2.0, [f])


def test_reshape(rng):
    f = rng.standard_normal((2, 3, 4))
    assert_gradient_correct(lambda ts: ts[0].reshape(2, 12).sum() * 3.0, [f])


def test_getitem_slice(rng):
    """The exact slicing pattern used to extract final-position logits."""
    x = rng.standard_normal((3, 4, 5))
    assert_gradient_correct(lambda ts: (lambda s: (s * s).sum())(ts[0][:, -1, :]), [x])


def test_embedding_lookup_gradient_and_repeated_index_accumulation():
    table = np.random.default_rng(1).standard_normal((10, 4))
    idx = np.array([2, 2, 5, 7])  # repeated index on purpose

    def loss_fn(table_arr):
        t = Tensor(table_arr, requires_grad=True)
        out = embedding_lookup(t, idx)
        loss = (out * out).sum()
        loss.backward()
        return loss.data, t.grad

    _, analytic = loss_fn(table.copy())
    num = np.zeros_like(table)
    eps = 1e-5
    it = np.nditer(table, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        orig = table[i]
        table[i] = orig + eps
        l1, _ = loss_fn(table.copy())
        table[i] = orig - eps
        l2, _ = loss_fn(table.copy())
        table[i] = orig
        num[i] = (l1 - l2) / (2 * eps)
    assert np.max(np.abs(num - analytic)) < TOL
    assert not np.allclose(analytic[2], 0), "repeated-index gradient should accumulate, not overwrite"


def test_cross_entropy_gradient():
    logits = np.random.default_rng(2).standard_normal((6, 5))
    targets = np.array([0, 1, 2, 3, 4, 0])

    def ce_loss(logits_arr):
        lt = Tensor(logits_arr, requires_grad=True)
        loss, probs = cross_entropy(lt, targets)
        loss.backward()
        return loss.data, lt.grad

    _, analytic = ce_loss(logits.copy())
    num = np.zeros_like(logits)
    eps = 1e-5
    it = np.nditer(logits, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        orig = logits[i]
        logits[i] = orig + eps
        l1, _ = ce_loss(logits.copy())
        logits[i] = orig - eps
        l2, _ = ce_loss(logits.copy())
        logits[i] = orig
        num[i] = (l1 - l2) / (2 * eps)
    assert np.max(np.abs(num - analytic)) < TOL


def test_cross_entropy_matches_known_value():
    """Sanity check against a hand-computed value: uniform logits over 4
    classes should give loss = ln(4)."""
    logits = Tensor(np.zeros((3, 4)))
    loss, probs = cross_entropy(logits, np.array([0, 1, 2]))
    assert abs(loss.item() - np.log(4)) < 1e-8
    assert np.allclose(probs, 0.25)


def test_adam_reduces_a_simple_quadratic():
    x = Tensor(np.array([5.0, -3.0]), requires_grad=True)
    opt = Adam([x], lr=0.1)
    for _ in range(200):
        loss = (x * x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.max(np.abs(x.data)) < 0.05


def test_backward_accumulates_not_overwrites_with_shared_subexpression():
    """x used twice in the same graph (x*x) must get BOTH contributions to
    its gradient, not just the last one visited."""
    x = Tensor(np.array([3.0]), requires_grad=True)
    y = x * x  # dy/dx should be 2x = 6, not x's single-use gradient of 3
    y.backward()
    assert np.allclose(x.grad, np.array([6.0]))
