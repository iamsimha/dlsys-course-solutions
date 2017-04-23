import numpy as np
import autodiff as ad

x = ad.Variable(name = "x")
w = ad.Variable(name = "w")
b = ad.Variable(name = "b")
labels = ad.Variable(name = "lables")

y = ad.matmul_op(w, x) + b

# loss = ad.reduce_sum(ad.log( 1.0 / 1 + ad.exp(-1.0 * labels * (ad.matmul_op(w, x) + b)) ))


# class_1 = np.random.normal(2, 0.1, (100, 2))
# class_2 = np.random.normal(4, 0.1, (100, 2))
# x_val = np.concatenate((class_1, class_2), axis = 0)
# y_val = np.concatenate((np.zeros_like(class_1), np.ones_like(class_2)), axis=0)
