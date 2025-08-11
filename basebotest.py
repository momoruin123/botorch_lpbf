from material_transfer import BaseBoClass

BO = BaseBoClass.BaseBoClass(2, 1)
BO.set_bounds([0, 5], [1, 6])
BO.run_bo()
