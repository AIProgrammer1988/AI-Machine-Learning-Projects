# Artificial Intelligence for Business
# Optimizing Warehouse Flows with Q-Learning

# Importing the Libraries

import numpy as np

# Setting the parameters gamma and alpha for the Q-Learning

gamma = 0.75
alpha = 0.9

# the lower alpha, the slower our machine will be learning Q-Learning

# Part 1: Defining the Environment (states, actions, and rewards)

# Defining the states with a Dictionary Key Value Pair (Location to Index)

location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11,}
    
# Defining the actions // List of all the actions that can be played overall
#actions are basically the indexes of the 12 locations

actions = [0,1,2,3,4,5,6,7,8,9,10,11]
    
# Defining the rewards // Create a matrix (state and action at specific time)
# Rows correspond to the states
# Columns correspond to the actions (0-11)
# Introducing a Two-D Array(Numpy) + [Rows] + [Columns]

R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0], #A
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0], #C
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0], #E
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1], #G
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0], #I
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]]) #K

              #A,B,C,D,E,F,G,H,I,J,K,L

# If want to take the route J -K over J-F, J-K needs to have a higher reward 
#than 
              # J-F, but lower than G because we still want the end point to be
              #G. 
              # We just want K to be the first checkpoint on the way to G.
              # Thus, make K have a reward of 500 and G a reward of 1000
              # Again, row is our current location and columnn is the next 
              #location.
              
              # Or add -500 going to J-F to the route we do not want
              # 


# Part 2: Building the solution with Q-Learning (Implement and Train Q)
              # Making a mapping from the states to the location


state_to_location = {state: location for location, 
                     state in location_to_state.items()}
    
    
# Shortest distance to optimal location with the highest reward value
# path composed of letters in the form of a list
# Starting point E, ending point of G. Inputs will be letters in quotes
# Optimal path from start to G
# Assume starting in Location E and final location to G
# Set high reward for specific location G
# To execute code, select everything,  command + enter
# Print Q Values directly from the consoles
# print(Q.astype(int))
    
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    Q = np.array(np.zeros([12,12]))
    # Initializing the Q-Value
    # Return a matrix of 12 rows and 12 columns full of zeros
    for i in range(1000):
        current_state = np.random.randint(0,12) # Next step is T+1
        playable_actions = [] # Empty List and Append the actions where R > 0
        for j in range(12):  # Go across all the 12 colmuns and inspect each
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions) # Create a random 
        #number from playable_actions
        # Choice Function will take a random element from a list # cell 
        #with highest Q-Value
        TD = R_new[current_state, next_state] + gamma * Q[next_state, 
                  np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, 
         next_state] + alpha * TD
    route = [starting_location]
    next_location = starting_location # dont know how many iterations 
    #it will take to go from starting to end, so while loop
    while (next_location != ending_location):
        #index of row then column (Row E , Colmun I or Index 4,8), 
        #Optimal Point
        # Row E - Then choose column with highest Q-Value
        # Use Dictionary to locate Key:Value Pair // Location_to_state Action 
        #We Play = Next State
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
         # Update starting location with the next_location so the loop 
         #continues
        starting_location = next_location
    return route
        
# Priority Locations: {1:G, 2:K, 3:l, 4:j, 5:A, 6:I,7:H, 8:C, 9:B, 10:D, 
#11:F, 12:E)}

# Part 3- Going into Production

def best_route(starting_location,intermediary_location,ending_location):
    return route(starting_location,intermediary_location) + route(
            intermediary_location,ending_location)[1:] 
    # Skips the first element 0 becausse indexing at 1
    # Duplicating K Index

# Printing the final route
print('Route:')
best_route('E', 'F', 'G')

        
# E, I, J, F, B, C, G; Route 1
# E, I, J, K, L, H, G; Route 2
    
    
    
    
    


