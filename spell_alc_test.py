import sys
import time
import argparse
from spell.fitting_alc import *

from spell.structures import solution2sparql, structure_from_owl
from spell.fitting import solve_incr, solve, mode



A1 = Structure(3,
      {
          "A" : {0,1},
          "B" : {0,2}
      },
      {i : {} for i in range(3)},{},{}
)
P1 = [0]
N1 = [1,2]

rn2= {i : {} for i in range(3)}
rn2[0] = {(2,'r')}
cn2 = {
          "A" : {0,1,2},
          "B" : {0,1}
      }
A2 = Structure(3, cn2,rn2, {},{})
P2 = [0]
N2 = [1]

A3 = Structure(3,
      {
          "A" : {1},
          "B" : {2}
      },
      {i : {} for i in range(3)},{},{}
)
P3 = [1,2]
N3 = [0]

rn4 = dict()
rn4[0] = {(0,'r')}
rn4[1] = {(2,'r')}
rn4[2] = {(3,'r')}
rn4[3] = {}
A4 = Structure(
    4,
    {
        "A":{0},
        "B":{3}
    }, rn4,   {},{}
)

P4 = [1]
N4 = [0]

A5 = Structure(
    2,
    {
        "A":{1},
        "B":{1}
    },{i : {} for i in range(2)},{},{}
               )
P5 = [0]
N5 = []

A6 = Structure(
    3,
    {
        "A":{1},
        "B":{0,1}
    },{i : {} for i in range(3)},{},{}
               )
P6 = [0]
N6 = [1]

def main():
    i1 = (A1,3,P1,N1)   
    i2 = (A2,3,P2,N2)      
    i3 = (A3,3, P3,N3)
    i4 = (A4,3,P4,N4)
    i5 = (A5, 1, P5, N5)
    i6 = (A6, 3, P6, N6)    
    f = FittingALC(*i1, op = {EX,ALL,OR,AND})
    f.solve()
    #f.printVariables()
    

if __name__ == "__main__":
    main()