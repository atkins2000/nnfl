from pso import *

random.seed(2)

# Given below are the sample test cases for all functions of pso. Uncomment whichever functions you want to test out.

# T1
# Sample test case for cost_function
X = [2, 1]
cost = cost_function(X)
assert cost == 5.0

# T2
# Sample test case for initialise
random.seed(2)
initial_position, initial_velocity, best_position, best_cost = initialise(3)
assert initial_position == [9.120685437784989, -8.868972645463826, 6.709977562588993]

# T3
# Sample test case for assess
position = [-1, 2, -3]
best_position = [2, 3, 4]
best_cost = -1
best_cost = assess(position, best_position, best_cost, lambda x: sum(x))
assert best_position == [-1, 2, -3]

# T4
# Sample test case for velocity_update
random.seed(2)
w = 0.2
c1 = 1
c2 = 2
velocity = [0.5, 0.5, 0.5]
position = [1, 2, 3]
best_position = [3, 4, 5]
best_group_position = [2, 3, 4]
velocity_update(w, c1, c2, velocity, position, best_position, best_group_position)
assert velocity == [3.907723517897198, 0.3828467257714606, 3.242937734395946]

# T5
# Sample test case for position_update
velocity = [0.5, 0.5, 0.5]
position = [1, 2, 3]
limits = [-10, 10]
position_update(position, velocity, limits)
assert position == [1.5, 2.5, 3.5]

# T6
# Sample test case for optimise
random.seed(2)
vector_length = 6
limits = [-10, 10]
w = 0.2
c1 = 1
c2 = 2
swarm_size = 15
max_iterations = 50
best_group_position, best_group_cost = optimise(vector_length, swarm_size, w, c1, c2, limits, max_iterations)
assert best_group_position == [-0.011339304431086255, -10.0, -0.00224581601324191, -9.036576634485082, -0.014853603978905627, 10.0]

