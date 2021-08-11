import random

# DO NOT import any other modules.
# DO NOT change the prototypes of any of the functions.
# Sample test cases given
# Grading will be based on hidden tests


# Cost function to be optimised
# Takes a list of elements
# Return the total sum of squares of even-indexed elements and inverse squares of odd-indexed elements
def cost_function(X): # 0.25 Marks
    # Your code goes here
    sum = 0
    for i in range(len(X)):
        if i%2==0:
            sum = sum + (X[i]*X[i])
        else:
            sum = sum + 1/(X[i]*X[i])
    return sum

# Takes length of vector as input
# Returns 4 values - initial_position, initial_velocity, best_position and best_cost
# Initialises position to a list with random values between [-10, 10] and velocity to a list with random values between [-1, 1]
# best_position is an empty list and best cost is set to -1 initially
def initialise(length): # 0.25 Marks
    # your code goes here
    best_position = [None]*length
    best_cost = -1
    initial_position = [None]*length
    initial_velocity = [None]*length
    for i in range(length):
        initial_position[i] = random.uniform(-10,10)
        initial_velocity[i] = random.uniform(-1,1)
    return initial_position, initial_velocity, best_position, best_cost   


# Evaluates the position vector based on the input func
# On getting a better cost, best_position is updated in-place
# Returns the better cost 
def assess(position, best_position, best_cost, func): # 0.25 Marks
    # Your code goes here
    if func(position)<best_cost or best_cost==-1:
        best_position[:] = position
        best_cost = func(position)
    return best_cost

# Updates velocity in-place by the given formula for each element:
# vel = w*vel + c1*r1*(best_position-position) + c2*r2*(best_group_position-position)
# where r1 and r2 are random numbers between 0 and 1 (not same for each element of the list)
# No return value
def velocity_update(w, c1, c2, velocity, position, best_position, best_group_position): # 0.5 Marks
    # Code goes here
    for i in range(len(position)):
        r1 = random.random()
        r2 = random.random()
        velocity [i] = w*velocity[i] + c1*r1*(best_position[i]-position[i]) + c2*r2*(best_group_position[i]-position[i])
    return None


# Input - position, velocity, limits(list of two elements - [min, max])
# Updates position in-place by the given formula for each element:
# pos = pos + vel
# Position element set to limit if it crosses either limit value
# No return value
def position_update(position, velocity, limits): # 0.5 Marks
    # Code goes here
    for i in range(len(position)):
        position[i] = position[i]+velocity[i]
        if position[i]>limits[1]:
            position[i] = limits[1]
        elif position[i]<limits[0]:
            position[i] = limits[0]
    return None


# swarm is a list of particles each of which is a list containing current_position, current_velocity, best_position and best_cost
# Initialise these using the function written above
# In every iteration for every swarm particle, evaluate the current position using the assess function (use the cost function you have defined) and update the particle's best cost if needed
# Update the best group cost and best group position based on performance of that particle
# Then for every swarm particle, first update its velocity then its position
# Return the best position and cost for the group
def optimise(vector_length, swarm_size, w, c1, c2, limits, max_iterations, initial_best_group_position=[], initial_best_group_cost=-1): # 1.25 Marks
    # Your Code goes here
    best_group_cost = initial_best_group_cost
    best_group_position = initial_best_group_position

    swarm = [list(initialise(vector_length)) for _ in range(swarm_size)]
    for _ in range(max_iterations):
        for particle in swarm:
            current_position, current_velocity, best_position, best_cost = particle[0], particle[1], particle[2], particle[3]
            current_cost = assess(current_position, best_position, best_cost, cost_function)
            if current_cost<best_cost or best_cost==-1: 
                best_cost = current_cost

            if current_cost<best_group_cost or best_group_cost==-1:
                best_group_position = best_position
                best_group_cost = best_cost
            particle[0],particle[1],particle[2],particle[3] = current_position, current_velocity, best_position, best_cost

        for particle in swarm:
            current_position, current_velocity, best_position, best_cost = particle[0],particle[1],particle[2],particle[3]
            velocity_update(w,c1,c2,current_velocity,current_position,best_position,best_group_position)
            position_update(current_position,current_velocity,limits)
            particle[0],particle[1],particle[2],particle[3] = current_position, current_velocity, best_position, best_cost
    
    return best_group_position, best_group_cost

