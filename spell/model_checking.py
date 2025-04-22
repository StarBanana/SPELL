from spell.fitting_alc import *
from spell.structures import Structure
CN = 7

class ALC_Concept():
    def __init__(self,sym,c1 = None, c2 = None, rn = None):
        self.sym = sym
        self.c1 = c1
        self.c2 = c2
        self.rn = rn

    def mc(self, A, a) -> bool:        
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
        assert False

    def __str__(self) -> str:
        if self.sym == TOP:
            return "TOP"
        if self.sym == BOT:
            return "BOT"
        if self.sym == CN:            
            return str(self.c1)
        if self.sym == AND:
            return "({} AND {})".format(self.c1, self.c2)
        if self.sym == OR:
            return "({} OR {})".format(self.c1, self.c2)
        if self.sym == NEG:
            return "NEG {}".format(self.c1)
        if self.sym == EX:                        
            return "∃.{} {}".format(self.rn, self.c1)
        if self.sym == ALL:
            return "∀.{} {}".format(self.rn, self.c1)
        assert False

def parse_string(inp: str) -> tuple[ALC_Concept, str]:
    inp = inp.lstrip()

    assert len(inp) > 0

    if inp.startswith("("):
        c1, inp = parse_string(inp[1:])
        inp = inp.lstrip()

        if inp.startswith("AND"):
            sym = AND
            inp = inp[4:]
        elif inp.startswith("OR"):
            sym = OR
            inp = inp[3:]
        else:
            assert False
        
        c2, inp = parse_string(inp)
        inp = inp.lstrip()

        assert inp[0] == ")"

        return ALC_Concept(sym, c1, c2), inp[1:]

    if inp.startswith("NEG "):
        c, inp = parse_string(inp[4:])
        return ALC_Concept(NEG, c), inp

    if inp.startswith("TOP"):
        return ALC_Concept(TOP), inp[3:]
    
    if inp.startswith("BOT"):
        return ALC_Concept(BOT), inp[3:]

    if inp.startswith("∀."):
        idx = inp.find(" ")
        rn = inp[2:idx]
        c, inp = parse_string(inp[idx + 1:])
        return ALC_Concept(ALL, c, rn=rn), inp
    
    if inp.startswith("∃."):
        idx = inp.find(" ")
        rn = inp[2:idx]
        c, inp = parse_string(inp[idx + 1:])
        return ALC_Concept(EX, c, rn=rn), inp

    idx = inp.find(" ")
    idx2 = inp.find(")")
    idx3 = len(inp)
    if idx2 != -1:
        idx3 = min(idx3, idx2)
    if idx != -1:
        idx3 = min(idx3, idx)

    return ALC_Concept(CN, inp[:idx3]), inp[idx3:]




def parse_concept(inp : str) -> ALC_Concept:
    c, s = parse_string(inp)
    print(str(c))
    print(s)
    assert len(s) == 0
    assert str(c) == inp
    return c


# print(str(parse_concept("∀.http://dl-learner.org/ont/hasScreening BOT")))

# cstr = "(http://www.example.org/lymphography#NON19_n0-9 OR ((http://www.example.org/lymphography#Bl_of_lymph_c4 OR http://www.example.org/lymphography#NON19_n10-19) AND (http://www.example.org/lymphography#CIN14_Lac_Margin OR http://www.example.org/lymphography#CIS15_Coarse)))"

# print(str(parse_concept(cstr)))
