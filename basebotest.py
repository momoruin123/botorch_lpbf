from material_transfer import base_bo_class

BO = base_bo_class.BaseBO(4, 3)
BO.read_train_data('./result/target_task_data.csv')
X, Y = BO.get_train_data()
print(X)
print(Y)
BO.set_bounds([0, 0, 0, 0], [200, 1000, 0.15, 0.5])
A = BO.set_ref_point(0.1)
print(A)
BO.run_bo()
