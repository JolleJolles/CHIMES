from ABSim.config import REFUGE_R, REFUGE_CX, REFUGE_CY, REFUGE_REPEL, REFUGE_BUFFER

class Circ_refuge:
    def __init__(self, x, y, r, repel, buff):
        self.x = x
        self.y = y
        self.r = r
        self.repel = repel # repel force
        self.buff = buff

cref = Circ_refuge(REFUGE_CX, REFUGE_CY, REFUGE_R, REFUGE_REPEL, REFUGE_BUFFER)