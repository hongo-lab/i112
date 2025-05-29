# -*- coding: utf-8 -*-
import numpy as np

def Gaussian_func(x, x0, y, y0, sx, sy):
    return 1000 * (1.0 / (2 * np.pi * sx * sy)) * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))

def dGaussian_func(x, x0, y, y0, sx, sy):
    common_exp = np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))
    dG_dx = 1000 * (-1.0 / (2 * np.pi * sy * (sx ** 3))) * (x - x0) * common_exp
    dG_dy = 1000 * (-1.0 / (2 * np.pi * sx * (sy ** 3))) * (y - y0) * common_exp
    return dG_dx, dG_dy

def func(x, y, x0a, y0a, sxa, sya):
    return sum([-1.0 * Gaussian_func(x, x0, y, y0, sx, sy) for x0, y0, sx, sy in zip(x0a, y0a, sxa, sya)])

def dfunc(x, y, x0a, y0a, sxa, sya):
    dfunc_dx_sum = 0.0
    dfunc_dy_sum = 0.0
    for x0, y0, sx, sy in zip(x0a, y0a, sxa, sya):
        dG_dx, dG_dy = dGaussian_func(x, x0, y, y0, sx, sy)
        dfunc_dx_sum += -1.0 * dG_dx
        dfunc_dy_sum += -1.0 * dG_dy
    return dfunc_dx_sum, dfunc_dy_sum

def out_field_data(file_name, x, y, x0a, y0a, sxa, sya):
    with open(file_name, "w") as f:
        for i in x:
            for j in y:
                f.write(f"{i} {j} {func(i, j, x0a, y0a, sxa, sya)}\n")
            f.write("\n")

x0 = [100, 100, 130, 150]  # Gaussian center
y0 = [105, 150, 130, 105]
sx = [13, 10, 10, 12]      # sigma
sy = [13, 10, 10, 12]

# Initial parameters
x_initial = 160.0
y_initial = 160.0
step_size = 1.0
cutoff = 1.0e-17

x_plot = np.arange(75, 175, 0.1)
y_plot = np.arange(80, 180, 0.1)

out_field_data("field.dat", x_plot, y_plot, x0, y0, sx, sy)

count = 0
x = x_initial
y = y_initial
z = func(x, y, x0, y0, sx, sy)

with open("data.dat", "w") as f:
    f.write(f"{count+1} {x:.17f} {y:.17f} {z:.17f}\n\n\n")
    while True:
        z_before = z

        df_dx, df_dy = dfunc(x, y, x0, y0, sx, sy)

        x -= step_size * df_dx
        y -= step_size * df_dy

        z = func(x, y, x0, y0, sx, sy)

        if count % 100 == 0:
            print(f'n = {count+1}, x = {x:.17f}, y = {y:.17f}, z = {z:.17f}')
            f.write(f"{count+1} {x:.17f} {y:.17f} {z:.17f}\n")
        if abs(z - z_before) < cutoff:
            break
        count += 1

    f.write(f"\n\n{count+1} {x:.17f} {y:.17f} {z:.17f}\n")
