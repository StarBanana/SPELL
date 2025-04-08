import time, os
from asciitree import LeftAligned
from asciitree.drawing import BoxStyle
from collections import OrderedDict as OD


from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose4 


from .structures import (
    Signature,
    Structure,
    restrict_to_neighborhood,
)

from .fitting import (
    determine_relevant_symbols,
)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

d_op = {
     0 : "TOP",
     1 : "BOT",
     2: "NEG",
     3: "AND",
     4: "OR",
     5: "EX",
     6: "ALL"
}
d_var_names = {
    0:"X",
    1:"Y",
    2:"Z",
    4:"V",
    5:"L"
}
TOP = 0
BOT = 1
NEG = 2
AND = 3
OR = 4
EX = 5
ALL = 6
ALC_OP = {NEG,AND,OR,EX,ALL}
ALC_OP_B = {NEG,AND,OR}
X = 0
Y = 1
Z = 2
V = 4
L = 5


TYPE_ENCODING: bool = True
NNF: bool = False

class STreeNode():
    def __init__(self, node, children):
        self.node = node
        self.children = children
    
    def _to_OD_c(self):
            if not self.children:
                return OD()
            else:
                return OD(list(map(lambda x : (x.node,x._to_OD_c()),self.children)))
            
    def _to_OD(self):                
        return OD([(self.node,self._to_OD_c())])

    def to_asciitree(self):
        tr = LeftAligned()
        tr.draw = BoxStyle(node_label = lambda x : x[1])
        return tr(self._to_OD())
    
    def to_string(self):                
        ns = os.path.basename(self.node[1])
        if len(self.children) == 0:
            return ns
        elif len(self.children) == 1:
            if self.node[1].startswith("all"):
                nss = f"∀.{ns}"
            elif self.node[1].startswith("ex"):
                nss = f"∃.{ns}"
            else:
                nss = ns
            return f"{nss} {self.children[0].to_string()}"
        elif len(self.children) == 2:
            return f"({self.children[0].to_string()} {self.node[1]} {self.children[1].to_string()})" 
        else:
            return ""
    
    @classmethod
    def FromDict(cls,dict,root):
        return cls(root, list(map(lambda x : cls.FromDict(dict, x), dict[root])))

class FittingALC:
    def __init__(self, A: Structure, k : int, P: list[int],
        N: list[int], op = ALC_OP, cov_p = - 1, cov_n = -1):
        B,m = restrict_to_neighborhood(k-1,A,P + N)
        self.P : list[int] = [m[a] for a in P]
        self.N : list[int] = [m[b] for b in N]
        self.A : Structure = B
        self.sigma : Signature = determine_relevant_symbols(A, P, 1, k - 1, N)
        self.k = k
        self.op = op
        self.op_b = ALC_OP_B.intersection(op)
        self.op_r = op.difference(ALC_OP_B)
        self.tree_node_symbols = dict()
        self.types = self.cn_types()
        self.vars = self._vars()
        self.n_op = len(op)        
        self.cov_p = len(P) if cov_p == -1 else cov_p
        self.cov_n = len(N) if cov_n == -1 else cov_n
        self.solver = Glucose4()
        self.max_var = 0
        
    def _vars(self):
        d = dict()
        i = 1
        d[X,TOP] = i
        self.tree_node_symbols[i] = d_op[TOP]
        d[X,BOT] = i * self.k + 1
        self.tree_node_symbols[i * self.k+1] = d_op[BOT]
        i+=1
        for cn in self.sigma[0]:
            d[X,cn] = i * self.k+1
            self.tree_node_symbols[i * self.k+1] = cn
            i += 1
        for op in self.op_b:
            d[X,op]=i * self.k+1
            self.tree_node_symbols[i * self.k+1] = d_op[op]
            i+=1
        if EX in self.op:
            for c in self.sigma[1]:
                d[X,EX,c] = i * self.k+1
                self.tree_node_symbols[i * self.k+1] = f"ex.{c}"
                i += 1
        if ALL in self.op:
            for c in self.sigma[1]:
                d[X,ALL,c] = i * self.k+1
                self.tree_node_symbols[i * self.k+1] = f"all.{c}"
                i+=1
        for l in range(self.k):
            d[Y,l] = i * self.k+1
            i +=1
        for a in range(self.A.max_ind):
            d[Z,a] = i * self.k+1
            i += 1
        for j in range(self.k):
            d[V,1,j] = i*self.k+1
            i+=1
        for j in range(self.k):
            d[V,2,j] = i*self.k+1
            i+=1

        if TYPE_ENCODING:
            for tp in self.types:
                d[X, tp] = i * self.k + 1
                i += 1

            # For leaves
            d[L] = i* self.k + 1
            i += 1

        self.max_var = i * self.k + 1000
        return d

    def _root(self):
        for j in range(1,self.k):
            self.solver.add_clause([-self.vars[Y,j]])

    def _syn_tree_encoding(self):
        for i in range(self.k):            
            x_vars = [self.vars[X,o]+i for o in self.op_b] + [self.vars[X,o,r]+i for o in self.op_r for r in self.sigma[1]] + [self.vars[X,cn] +i for cn in self.sigma[0]] + [self.vars[X,TOP]+i,self.vars[X,BOT]+i]
            self.solver.add_clause(x_vars)
            for v1 in x_vars:
                for v2 in x_vars:
                    if v1 != v2:
                        self.solver.add_clause((-v1,-v2))

        for i in range(self.k):
            v_vars = [self.vars[V, 1, i] + j for j in range(i + 1, self.k)] + [self.vars[V, 2, i] + j for j in range(i + 1, self.k)]

            # At most one of the y-vars
            for v1 in v_vars:
                for v2 in v_vars:
                    if v1 != v2:
                        self.solver.add_clause((-v1,-v2))

            for r in self.sigma[1]:
                for op in self.op_r:
                    self.solver.add_clause([-(self.vars[X,op,r]+i)] + [self.vars[V,1,i]+j for j in range(i+1,self.k)])
                    for j in range(self.k):
                        self.solver.add_clause([-(self.vars[X,op,r]+i), -(self.vars[V,2,i]+j)])

            if NEG in self.op_b:
                self.solver.add_clause([-(self.vars[X,NEG]+i)] + [self.vars[V,1,i]+j for j in range(i+1,self.k)])
                for j in range(self.k):
                    self.solver.add_clause([-(self.vars[X,NEG]+i), -(self.vars[V,2,i]+j)])
            for op in self.op_b - {NEG}:
                self.solver.add_clause([-(self.vars[X,op]+i)] + [self.vars[V,2,i]+j for j in range(i+1,self.k-1)])
                for j in range(self.k):
                    self.solver.add_clause([-(self.vars[X,op]+i), -(self.vars[V,1,i]+j)])

            for cn in self.sigma[0]:

                if TYPE_ENCODING:
                    # Is a leaf
                    self.solver.add_clause((-(self.vars[X,cn]+i),(self.vars[L] + i)))

            for j in range(self.k):
                for cn in self.sigma[0]:
                    self.solver.add_clause((-(self.vars[X,cn]+i),-(self.vars[V,1, i]+j)))
                    self.solver.add_clause((-(self.vars[X,cn]+i),-(self.vars[V,2, i]+j)))


                for b in {TOP,BOT}:
                    self.solver.add_clause((-(self.vars[X,b]+i),-(self.vars[V,1,i]+j)))
                    self.solver.add_clause((-(self.vars[X,b]+i),-(self.vars[V,2,i]+j)))




            for j1 in range(self.k):
                for j2 in range(self.k):
                    # Just one predecessor
                    if j1 != j2:
                        self.solver.add_clause((-(self.vars[V,1,j1]+i),-(self.vars[V,1,j2]+i)))
                        self.solver.add_clause((-(self.vars[V,1,j1]+i),-(self.vars[V,2,j2]+i)))
                        self.solver.add_clause((-(self.vars[V,1,j1]+i),-(self.vars[V,2,j2]+i - 1)))

                        self.solver.add_clause((-(self.vars[V,2,j1]+i),-(self.vars[V,1,j2]+i)))
                        self.solver.add_clause((-(self.vars[V,2,j1]+i),-(self.vars[V,2,j2]+i - 1)))
                        self.solver.add_clause((-(self.vars[V,2,j1]+i),-(self.vars[V,2,j2]+i)))


    def _symmetry_breaking(self):
        # Symmetry breaking: crossing free syntax tree
        for i in range(self.k):
            for j in range(i + 1, self.k):
                for i2 in range(i + 1, j):
                    for j2 in range(j + 1, self.k):
                        self.solver.add_clause((-(self.vars[V,1,i]+j),-(self.vars[V,1,i2]+j2)))
                        self.solver.add_clause((-(self.vars[V,1,i]+j),-(self.vars[V,2,i2]+j2)))
                        self.solver.add_clause((-(self.vars[V,2,i]+j),-(self.vars[V,1,i2]+j2)))
                        self.solver.add_clause((-(self.vars[V,2,i]+j),-(self.vars[V,2,i2]+j2)))

        # Symmetry breaking: associativity of sqcap and sqcup
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if AND in self.op_b:
                    self.solver.add_clause( (- (self.vars[X, AND] + i), - (self.vars[V, 2, i] + j), - (self.vars[X, AND] + j)))
                if OR in self.op_b:
                    self.solver.add_clause( (- (self.vars[X, OR] + i), - (self.vars[V, 2, i] + j), - (self.vars[X, OR] + j)))

        # Symmetry breaking: limited commutativity
        for i in range(self.k):
            for j in range(i + 1, self.k):
                # This orders the node types. Binary operators are smallest, the unary operators, then conceptnames, then top and bot
                x_varsj = [self.vars[X,o]+j for o in self.op_b] + [self.vars[X,o,r]+j for o in self.op_r for r in self.sigma[1]] + [self.vars[X,cn] +j for cn in self.sigma[0]] + [self.vars[X,TOP]+j,self.vars[X,BOT]+j]
                x_varsj1 = [self.vars[X,o]+j + 1 for o in self.op_b] + [self.vars[X,o,r]+j + 1 for o in self.op_r for r in self.sigma[1]] + [self.vars[X,cn] +j + 1 for cn in self.sigma[0]] + [self.vars[X,TOP]+j + 1,self.vars[X,BOT]+j + 1]
                assert(len(x_varsj) == len(x_varsj1))
                for k1 in range(len(x_varsj)):
                    for k2 in range(k1 + 1, len(x_varsj1)):
                        # left must be "bigger" than right
                        self.solver.add_clause( ( - (self.vars[V, 2, i] + j), - x_varsj[k1], -x_varsj1[k2]))
                        # Note: This should not conflict with the associativity breaking above: binary operators should be always allowed to the right! child


        # NNF    
        if NNF:
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    for j2 in range(j + 1, self.k):
                        self.solver.add_clause (( - (self.vars[X, NEG] + i),   - (self.vars[V, 1, i] + j), - (self.vars[V, 1, j] + j2)))
                        self.solver.add_clause (( - (self.vars[X, NEG] + i),   - (self.vars[V, 1, i] + j), - (self.vars[V, 2, j] + j2)))



    def _evaluation_constraints(self):

        for a in range(self.A.max_ind):
            for i in range(self.k):                
                if NEG in self.op_b:
                    for j in range(i + 1, self.k):
                        self.solver.add_clause((-(self.vars[X,NEG]+i), -(self.vars[Z,a]+i), -(self.vars[V,1,i]+j), - (self.vars[Z, a] + j)))
                        self.solver.add_clause((-(self.vars[X,NEG]+i), (self.vars[Z,a]+i), -(self.vars[V,1,i]+j), (self.vars[Z, a] + j)))

                if AND in self.op_b:
                    for j in range(i + 1, self.k - 1):
                        self.solver.add_clause((-(self.vars[X,AND]+i), -(self.vars[Z,a]+i), -(self.vars[V,2,i]+j), self.vars[Z, a] + j))
                        self.solver.add_clause((-(self.vars[X,AND]+i), -(self.vars[Z,a]+i), -(self.vars[V,2,i]+j), self.vars[Z, a] + j + 1))
                        self.solver.add_clause((-(self.vars[X,AND]+i), (self.vars[Z,a]+i), -(self.vars[V,2,i]+j), -(self.vars[Z, a] + j + 1), -(self.vars[Z, a] + j)))

                if OR in self.op_b:
                    for j in range(i + 1, self.k - 1):
                        self.solver.add_clause((-(self.vars[X,OR]+i), (self.vars[Z,a]+i), -(self.vars[V,2,i]+j), -(self.vars[Z, a] + j)))
                        self.solver.add_clause((-(self.vars[X,OR]+i), (self.vars[Z,a]+i), -(self.vars[V,2,i]+j), -(self.vars[Z, a] + j + 1)))
                        self.solver.add_clause((-(self.vars[X,OR]+i), -(self.vars[Z,a]+i), -(self.vars[V,2,i]+j), (self.vars[Z, a] + j + 1), (self.vars[Z, a] + j)))
                
                if ALL in self.op_r:
                    for r in self.sigma[1]:
                        for j in range(i + 1, self.k):
                            self.solver.add_clause([-(self.vars[X,ALL,r]+i), (self.vars[Z,a]+i), -(self.vars[V, 1, i] + j)] + [ -(self.vars[Z, b]+j) for b in map(lambda x : x[0], filter(lambda t : t[1] == r , self.A.rn_ext[a])) ])                            
                            for b in map(lambda t : t[0],filter(lambda t : t[1] == r , self.A.rn_ext[a])):
                                self.solver.add_clause((-(self.vars[X,ALL,r]+i), -(self.vars[Z,a]+i), -(self.vars[V, 1, i] + j), self.vars[Z, b] + j))

                if EX in self.op_r:
                    for r in self.sigma[1]:
                        for j in range(i + 1, self.k):
                            self.solver.add_clause([-(self.vars[X,EX,r]+i), -(self.vars[Z,a]+i), -(self.vars[V, 1, i] + j)] + [ (self.vars[Z, b]+j) for b in map(lambda x : x[0], filter(lambda t : t[1] == r , self.A.rn_ext[a])) ])                            
                            for b in map(lambda t : t[0],filter(lambda t : t[1] == r , self.A.rn_ext[a])):
                                self.solver.add_clause((-(self.vars[X,EX,r]+i), (self.vars[Z,a]+i), -(self.vars[V, 1, i] + j), -(self.vars[Z, b] + j)))

                self.solver.add_clause((-(self.vars[X,TOP]+i),(self.vars[Z,a]+i)))
                self.solver.add_clause((-(self.vars[X,BOT]+i),-(self.vars[Z,a]+i)))


        if not TYPE_ENCODING:
            for cn in self.sigma[0]:
                for i in range(self.k):
                    for a in range(self.A.max_ind):                    
                        if a in self.A.cn_ext[cn]:                                            
                            self.solver.add_clause((-(self.vars[X,cn]+i), self.vars[Z,a]+i))
                        else:
                            self.solver.add_clause((-(self.vars[X,cn]+i),-(self.vars[Z,a]+i)))

        if TYPE_ENCODING:
            for i in range(self.k):
                for tp in self.types:
                    for cn in self.sigma[0]:
                        if cn in tp:
                            self.solver.add_clause((-(self.vars[X, cn] + i), self.vars[X, tp] + i))
                        if cn not in tp: 
                            self.solver.add_clause((-(self.vars[X, cn] + i), -(self.vars[X, tp] + i)))


            for a in range(self.A.max_ind):
                tp = frozenset({ cn for cn in self.sigma[0] if a in self.A.cn_ext[cn]})
                assert tp in self.types
                for i in range(self.k):
                    self.solver.add_clause( ( - (self.vars[X, tp] + i),   self.vars[Z, a] + i))
                    # Problem: the following should only happen for CONCEPT NAME NODES. We thus need an additional variable that is true iff a node is a concept name node
                    self.solver.add_clause( ( (self.vars[X, tp] + i) ,   - (self.vars[Z, a] + i), - (self.vars[L] + i)))
                


    def _fitting_constraints(self):
        for a in self.P:
            self.solver.add_clause([self.vars[Z,a]])            
        for a in self.N:
            self.solver.add_clause([-(self.vars[Z,a])])       

    def _fitting_constraints_approximate(self, k: int):
        lits = [self.vars[Z, a] for a in self.P] + [-self.vars[Z, b] for b in self.N]
        
        # TODO maybe switch to incremental totalizer encoding or another incremental encoding
        enc = CardEnc.atleast(
            lits, bound=k, top_id=self.max_var, encoding=EncType.totalizer
        )
        for clause in enc.clauses:
            self.solver.add_clause(clause)

    def solve(self):
        acc, n, sol = self.solve_incr(self.k, self.k)
        if sol:
            return True
        else:
            return False
    
    def solve_incr(self,max_k :int, start_k : int =1, return_string = False):
        return self.solve_incr_approx(max_k, start_k, len(self.P) + len(self.N), -1)

    def cn_types(self) -> set[frozenset[str]]:
        res: set[frozenset[str]] = set()
        for i in range(self.A.max_ind):
            tp: list[str] = []
            for cn in self.sigma[0]:
                if i in self.A.cn_ext[cn]:
                    tp.append(cn)
            res.add(frozenset(tp))
        return res
                
    def solve_approx(self, k: int, min_n: int, timeout : float = -1):
        time_start = time.process_time()
        self.k = k
        n = max(len(self.P), len(self.N), min_n)
        
        dt = time.process_time() - time_start

        best_sol = None
        best_accuracy = 0
        best_n = 0

        while n <= len(self.P) + len(self.N) and (dt < timeout or timeout == -1):
            self.solver = Glucose4()
            self.vars = self._vars()
            self._syn_tree_encoding()
            self._evaluation_constraints()
            self._symmetry_breaking()
            self._fitting_constraints_approximate(n)

            if not self.solver.solve():
                print(f"Not satisfiable for k={self.k}, n={n}")
                return best_accuracy, best_n, self.k, best_sol

            best_sol = self._modelToTree()

            model_n = self._model_n()

            best_accuracy = model_n / (len(self.P) + len(self.N))
            best_n = model_n
            print(f"Satisfiable for k={self.k}, n={model_n}, acc={best_accuracy}")
            print(best_sol.to_asciitree())
            n = model_n + 1
        
        return best_accuracy, best_n, self.k, best_sol


    def solve_incr_approx(self, max_k : int , start_k : int =1, min_n: int = 1, timeout : float = -1):
        time_start = time.process_time()
        self.k = start_k
        self.k = 5
        n = max(len(self.P), len(self.N), min_n)
        best_sol = None
        best_acc = 0
        dt = time.process_time() - time_start

        while self.k <= max_k and (dt < timeout or timeout == -1):
            remaining_time = -1
            if timeout != -1:
                remaining_time = timeout - dt
            k_acc, k_n, _, k_sol = self.solve_approx(self.k, n, remaining_time)

            if k_acc > best_acc:
                assert k_sol
                best_sol = k_sol
                best_acc = k_acc
                n = k_n + 1

            self.k += 1 
            dt = time.process_time() - time_start

        if best_sol:
            return best_acc, self.k, best_sol
        else:
            return 0,-1,""

    def printVariables(self):
        if self.solver.get_model():
            l = self.solver.get_model()                
            for k,v in self.vars.items():
                for i in range(self.k):
                    s = f"{bcolors.FAIL}False{bcolors.ENDC}"
                    if v+i in l:
                        s = f"{bcolors.OKGREEN}True{bcolors.ENDC}" 
                    if k[0] == X:
                        try:
                            print((d_var_names[k[0]],d_op[k[1]],v+i), f"Tree Node: {(v+i-1)%self.k }",s)
                        except KeyError:
                            print((d_var_names[k[0]],k[1],v+i), f"Tree Node: {(v+i-1)%self.k }",s)
                    elif k[0] == V:
                        print((d_var_names[k[0]],k[1:],f"edge: ({k[2]},{i})",v+i),s)
                    else:
                        print((d_var_names[k[0]],k[1:],v+i),s)
                    if k[0] == Y:
                        print((d_var_names[k[0]],k[1:],v+i),s)

    def _model_n(self)-> int:
        # Return the number of positive/negative examples that is claimed to be covered by a model
        m = self.solver.get_model()

        res: int = 0
        for p in self.P:
            if self.vars[Z, p] + 0 in m:
                res += 1
        
        for n in self.N:
            if self.vars[Z, n] + 0 not in m:
                res += 1
        return res


    def _modelToTree(self) -> STreeNode:
            m = self.solver.get_model()
            edges = {i : [] for i in range(self.k)}
            x_symbols = [None] * self.k

            for x in m[:self.vars[Z,0]-1]:
                if x>0:
                    i = (x-1)%(self.k)                
                    x_symbols[i] = self.tree_node_symbols[x - i]

            for i in range(self.k):
                for j in range(i + 1, self.k):
                    if (self.vars[V, 1, i] + j) in m:
                        edges[i].append(j)
                    elif (self.vars[V, 2, i] + j) in m:
                        edges[i].append(j)
                        edges[i].append(j + 1)

            edges_labeled = { (k,x_symbols[k]) : list(map(lambda x : (x,x_symbols[x]), v)) for k,v in edges.items() }        
            t = STreeNode.FromDict(edges_labeled,(0,x_symbols[0]))        
            return t
