# ---------------------
# Basic setup parameters
# ---------------------

# Number of agents to be generated
NUM_BOIDS = 100
NUM_PREDS = 5

# Scenario toggles
# These control which behavior rules will be applied. Default all False, planning to code an option to change them upon running the program
ENABLE_SOCIAL = True               # Boid social rules (centering, avoidance, velocity matching)
ENABLE_CLASS_DIFF = True            # If false, all agent-types have the same behavioural rules
ENABLE_PREDATION = True             # Predation system (hunting, fleeing, prey detection) -- design note: if FALSE, predators would behave just like boids
# ENABLE_SIGNALING = True             # Predator signaling (predators communicate with each other) -- excluded as it's messy as is and not necessary for a core model



## CANVAS CONFIGURATION
WIDTH = 1920
HEIGHT = 1080
MARGIN = 100

## BOID CONFIGURATION
BOID_VIS_RANGE = 75
SPEED_LIMIT = 10
TURN_FACTOR = 3 # for turning away from the arena walls
BOID_CENTERING_FACTOR = 0.005
AVOID_FACTOR = 0.05
MATCHING_FACTOR = 0.05
DFLT_AVOID_DIST = 10
DRAW_TRAIL = False
DFLT_FLEE_DIST = 100
FLEE_FACTOR = 1.5

"""Trying to bind some variables to Tkinter widgets
added a live version of AVOID_DIST to within main()"""

## PREDATOR CONFIGURATION
HUNTING_SIGHT = 100
PRED_COHESION_VIS_RANGE = 100
PRED_CENTERING_FACTOR = 0.005
#HEARING_RANGE = HUNTING_SIGHT*4
DFLT_PRED_AVOID_DIST = 30
PRED_REP_HUNT_FACTOR = 0.2
PRED_REP_WANDER_FACTOR = 0.5
CATCH_DIST = 40 

## REFUGE CONFIGURATION -- Defining it as a circular refuge first
REFUGE_R = 150
REFUGE_CX = 400
REFUGE_CY = 300
REFUGE_REPEL = 20
REFUGE_BUFFER = 50

