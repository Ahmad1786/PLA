
import numpy as np
import matplotlib.pyplot as plt
import random

num_experiments = 1000
N = 10 # number of data points

def target_func(X, Y, m, b):
    if Y > m*X + b:
        return 1
    elif Y < m*X + b:
        return -1

s = 0
s2 = 0
for i in range(num_experiments):   
    # create one line
    x1, y1 = np.random.uniform(-1, 1, size=2)
    x2, y2 = np.random.uniform(-1, 1, size=2)
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m*x2
    x_values = np.linspace(-1, 1, 100)
    y_values = m * x_values + b

    random_points = np.random.uniform(-1, 1, size=(N, 2))
    weight_vector = np.array(np.zeros(3))

    solved = False
    iterations = 0
    while not solved:
        missclassed_points = []
        iterations += 1
        for arr in random_points:
            x, y = arr[0], arr[1]
            x_vector = [1, x, y]
            x_vector = np.array(x_vector)
            sign = np.sign(np.dot(weight_vector, x_vector))
            correct_sign = target_func(x, y, m, b)
            if sign != correct_sign:
                missclassed_points.append((x, y))

        if missclassed_points:
            missed_point = random.choice(missclassed_points)
            weight_vector = weight_vector + target_func(missed_point[0], missed_point[1], m, b)*np.array([1, missed_point[0], missed_point[1]])
        else:
            solved = True
            
            s += iterations
            diff = 0
            num_points = 500
            # create new points
            random_points2 = np.random.uniform(-1, 1, size=(num_points, 2))

            #for each get target value get sign, get ip get sign, compare and diff +1 if diff
            for arr in random_points2:
                target_sign = target_func(arr[0], arr[1], m, b)
                xvec = np.array([1, arr[0], arr[1]])
                sign2 = np.sign(np.dot(weight_vector, xvec))
                if sign2 != target_sign:
                    diff += 1
            s2 += diff/num_points

print(f"Iterations to converge: {s/num_experiments}") 
print(f"Generalization error : {s2/num_experiments}")         

