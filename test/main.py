from pymoo.problems import get_problem

problem = get_problem("zdt1", n_var=10)
print(problem.xl)   # 下界
print(problem.xu)   # 上界
