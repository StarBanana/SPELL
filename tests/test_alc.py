import sys
import time
import argparse
from spell.fitting_alc import *

from spell.structures import solution2sparql, structure_from_owl
from spell.fitting import solve_incr, solve, mode


def test1():
    A1 = Structure(3,
          {
              "A" : {0,1},
              "B" : {0,2}
          },
          {i : {} for i in range(3)},{},{}
    )
    P1 = [0]
    N1 = [1,2]
    i = (A1,3,P1,N1)   
    f = FittingALC(*i, op = {EX,ALL,OR,AND})
    assert f.solve()


def test2():
    rn2= {i : {} for i in range(3)}
    rn2[0] = {(2,'r')}
    cn2 = {
              "A" : {0,1,2},
              "B" : {0,1}
          }
    A2 = Structure(3, cn2,rn2, {},{})
    P2 = [0]
    N2 = [1]
    i = (A2,3,P2,N2)   
    f = FittingALC(*i, op = {EX,ALL,OR,AND})
    assert f.solve()

def test3():
    A3 = Structure(3,
          {
              "A" : {1},
              "B" : {2}
          },
          {i : {} for i in range(3)},{},{}
    )
    P3 = [1,2]
    N3 = [0]
    i = (A3,3,P3,N3)   
    f = FittingALC(*i, op = {EX,ALL,OR,AND})
    assert f.solve()


def test4():
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
    i = (A4,3,P4,N4)   
    f = FittingALC(*i, op = {EX,ALL,OR,AND})
    assert f.solve()


def test5():
    A5 = Structure(
        2,
        {
            "A":{1},
            "B":{1}
        },{i : {} for i in range(2)},{},{}
                   )
    P5 = [0]
    N5 = []
    i = (A5,1,P5,N5)   
    f = FittingALC(*i, op = {EX,ALL,OR,AND})
    assert f.solve()


def test6():
    A6 = Structure(
        3,
        {
            "A":{1},
            "B":{0,1}
        },{i : {} for i in range(3)},{},{}
                   )
    P6 = [0]
    N6 = [1]
    i = (A6,2,P6,N6)
    f = FittingALC(*i, op = {EX,ALL,OR,AND,NEG})
    assert f.solve()

def testEx():
    A = Structure(
        5,
        {"A":{2, 4, 5}, "B":{3}},
        { 0 : { (2, "r"), (3, "r")}, 1: {(4, "r"), (5, "r")}, 2: {}, 3: {}, 4: {}, 5:{}},
        {},
        {}
    )

    i = (A, 2, [0], [1])
    f = FittingALC(*i, op = {EX})
    assert f.solve()
    
    i2 = (A, 2, [1], [0])
    f2 = FittingALC(*i2, op = {EX})
    assert not f2.solve()

def testAnd():
    A = Structure(
        5,
        {"A":{1, 2, 3}, "B":{1, 3, 4}, "C":{1, 2, 4}},
        { 0 : set(), 1: set(), 2: set(), 3:set(), 4: set()},
        {},
        {}
    )

    i = (A, 4, [1], [2, 3, 4])
    f = FittingALC(*i, op = {AND})
    assert not f.solve()
    
    i2 = (A, 5, [1], [2, 3, 4])
    f2 = FittingALC(*i2, op = {AND})
    assert f2.solve()


def testAll():
    A = Structure(
        5,
        {"A":{2, 4, 5}, "B":{3}},
        { 0 : { (2, "r"), (3, "r")}, 1: {(4, "r"), (5, "r")}, 2: {}, 3: {}, 4: {}, 5:{}},
        {},
        {}
    )

    i = (A, 2, [0], [1])
    f = FittingALC(*i, op = {ALL})
    assert not f.solve()
    
    i2 = (A, 2, [1], [0])
    f2 = FittingALC(*i2, op = {ALL})
    assert f2.solve()


def testSize():
    k = 15
    # TODO: the SAT formula for this takes a surprising amount of time to solve
    # i.e. it is not instant
    # I believe if we modify our encoding such that this becomes instant, we can gain
    # a lot of speed on realistic benchmarks
    A = Structure(
        max_ind=k,
        cn_ext={},
        rn_ext= { i : {(i + 1, "r")} for i in range(k - 1)},
        indmap={},
        nsmap={}
    )
    A.rn_ext[k - 1] = set()

    i = (A, k, [0], [1])
    f = FittingALC(*i, op= {EX, AND})
    assert f.solve()