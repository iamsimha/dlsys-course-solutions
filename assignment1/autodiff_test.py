import autodiff as ad
import numpy as np

def test_identity():
    x2 = ad.Variable(name = "x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_identity_simple():
    x2 = ad.Variable(name = "x2")
    y = x2 + 3
    executor = ad.Executor([y])
    x2_val = 2 * np.ones(3)
    y_val, = executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 3)

def test_add_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))

def test_mul_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 * x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)

def test_add_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + x3_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
    assert np.array_equal(grad_x3_val, np.ones_like(x3_val))

def test_mul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)

def test_add_mul_mix_1():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x1 + x2 * x3 * x1

    grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val * x1_val)
    assert np.array_equal(grad_x3_val, x2_val * x1_val)

def test_add_mul_mix_2():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    x4 = ad.Variable(name = "x4")
    y = x1 + x2 * x3 * x4

    grad_x1, grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x1, x2, x3, x4])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val, x4 : x4_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
    assert np.array_equal(grad_x2_val, x3_val * x4_val)
    assert np.array_equal(grad_x3_val, x2_val * x4_val)
    assert np.array_equal(grad_x4_val, x2_val * x3_val)

def test_add_mul_mix_3():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    z_val = x2_val * x2_val + x2_val + x3_val + 3
    expected_yval = z_val * z_val + x3_val
    expected_grad_x2_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
    expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)

def test_grad_of_grad():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 * x2 + x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    grad_x2_x2, grad_x2_x3 = ad.gradients(grad_x2, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = x2_val * x2_val + x2_val * x3_val
    expected_grad_x2_val = 2 * x2_val + x3_val
    expected_grad_x3_val = x2_val
    expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
    expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
    assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
    assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)

def test_matmul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = ad.matmul_op(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]]) # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)

def test_mat_add():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    y = ad.matadd_op(2 * x1, x2)
    grad_y_x1, = ad.gradients(y, [x1])
    executor = ad.Executor([y, grad_y_x1])

    x1_val = np.ones((10, 10))
    x2_val = 2 * np.ones((10, 10))
    y_val, grad_y_x1_val = executor.run(feed_dict={x1:x1_val, x2:x2_val})
    assert np.array_equal(y_val, 4 * np.ones((10, 10)))
    assert np.array_equal(grad_y_x1_val, 2 * np.ones((10, 10)))

def test_log():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    y = ad.log(x1) / x2
    grad_y, = ad.gradients(y, [x1])

    x1_val = 2 * np.ones((2, 1))
    x2_val = np.ones((1,1))
    executor = ad.Executor([y, grad_y])

    y_val, grad_y_val, = executor.run(feed_dict = {x1: x1_val, x2:x2_val})
    assert np.array_equal(y_val, np.log(x1_val))
    assert np.array_equal(grad_y_val, 0.5 * np.ones_like(grad_y_val))

def test_exp():
    x1 = ad.Variable(name = "x1")
    y = 1 + 2 * ad.exp(ad.log(x1))

    x1_val = np.ones((2, 1))
    grad_y, = ad.gradients(y, [x1])
    executor = ad.Executor([y, grad_y])
    y_val, grad_y_val, = executor.run(feed_dict = {x1: x1_val})
    assert np.array_equal(y_val, 3 * np.ones_like(y_val))
    assert np.array_equal(grad_y_val, 2 * np.ones_like(grad_y_val))

test_mat_add()