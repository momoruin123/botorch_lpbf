from botorch.sensitivity import SobolSensitivity

sobol = SobolSensitivity(model=gp1, bounds=input_bounds)
indices = sobol.compute()
