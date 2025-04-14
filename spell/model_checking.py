from spell.fitting_alc import *
from spell.structures import Structure
CN = 7

class ALC_Concept():
    def __init__(self,sym,c1 = None, c2 = None, rn = None):
        self.sym = sym
        self.c1 = c1
        self.c2 = c2
        self.rn = rn

    def mc(self, A, a):        
        if self.sym == TOP:
            return True
        if self.sym == BOT:
            return False
        if self.sym == CN:            
            return a in A.cn_ext[self.c1]        
        if self.sym == AND:
            return self.c1.mc(A,a) and self.c2.mc(A,a)
        if self.sym == OR:
            return self.c1.mc(A,a) or self.c2.mc(A,a)
        if self.sym == NEG:
            return not self.c1.mc(A,a)
        if self.sym == EX:                        
            return any(map(lambda b : self.c1.mc(A,b[0]), filter(lambda t : t[1] == self.rn,A.rn_ext[a])))
        if self.sym == ALL:
            return all(map(lambda b : self.c1.mc(A,b[0]), filter(lambda t : t[1] == self.rn,A.rn_ext[a])))