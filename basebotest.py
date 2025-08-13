from material_transfer import base_bo_class, warm_start_bo_class


# ==== test warm start bo
wsbo = warm_start_bo_class.WarmStartBO(5, 2)

wsbo.set_bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])

X_src, Y_src = wsbo.read_data('./data/source_task_data.csv')
wsbo.add_source_data(X_src, Y_src)
print(wsbo.X_src.shape)
print(wsbo.Y_src.shape)

X, Y = wsbo.read_data('./data/target_task_data.csv')
wsbo.add_data(X, Y)
print(wsbo.X.shape)
print(wsbo.Y.shape)

wsbo.set_ref_point(2)
print(wsbo.get_ref_point())

wsbo.run_bo()
# ====test base bo
# BO = base_bo_class.BaseBO(4, 3)
#
# BO.read_data('./result/target_task_data.csv')
# X, Y = BO.get_train_data()
# # print(X)
# # print(Y)
# BO.set_bounds([0, 0, 0, 0], [200, 1000, 0.15, 0.5])
# A = BO.set_ref_point(0.1)
# print(A)
# BO.run_bo()

