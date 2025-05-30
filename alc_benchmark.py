import sys, random, json, os, time
import pandas as pd, owlapy as ow
from pathlib import Path
from rdflib import Graph
from spell.benchmark_tools import construct_owl_from_structure
from spell.fitting_alc import ALL, AND, EX, OR, NEG, FittingALC
from spell.structures import map_ind_name, restrict_to_neighborhood, structure_from_owl
from owlready2 import default_world, get_ontology, owl
from ontolearn_benchmark import run_evo
import spell.fitting_alc1 as fitting_alc1
import subprocess


CELOE_PATH = ""
SPARCEL_PATH = ""

RANDOM_SEED = 1

QALL = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    }
    """

random.seed(RANDOM_SEED)

def query_and_print(path, query):
    g = Graph()
    with open(path, 'r') as f:        
        g.parse(f, format='application/rdf+xml')
    qres = g.query(query)  
    for res in list(qres):
        print(res[0].toPython())

def query_for_examples(kb_path, q_pos, q_neg, n_pos, n_neg):
    g = get_ontology(kb_path).load()    
    p_res = list( map(lambda x : x[0].get_iri(), default_world.sparql(q_pos)))    
    P = random.sample(p_res,n_pos)    
    n_res = list(map(lambda x : x[0].get_iri(), default_world.sparql(q_neg)))
    N = random.sample(n_res,n_neg)      
    return P,N

def read_examples_from_json(path):
    with open(path) as f:
        o = json.load(f)
    return o["P"],o["N"]

def query_and_save(path, q_pos, q_neg, n_pos, n_neg, dest_path, filename):
    d = dict()
    P,N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    d["Q_POS"] = q_pos
    d["Q_NEG"] = q_pos
    d["N_POS"] = len(P)
    d["N_NEG"] = len(N)
    d["P"] = P
    d["N"] = N    
    with open(os.path.join(f"{dest_path}", f"{filename}.json"), 'w+') as f:
        json.dump(d,f, indent = 4)

def query_and_solve(path, q_pos, q_neg, n_pos, n_neg, k):
    P,N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND,NEG})
    return f.solve()

def instance_to_sparcel(kb_path, p, n, dest, file_name = "dl_instance"):
    file = os.path.join(dest,f"{file_name}.conf")
    with open(file, "w+", encoding="utf-8") as f:
        f.write('ks.type = "OWL File"\n')
        f.write(f'ks.fileName = "{os.path.relpath(kb_path)}"\n')
        f.write('reasoner.type = "fast instance checker"\n')
        f.write('reasoner.sources = { ks }\n')
        f.write('lp.type = "org.dllearner.algorithms.ParCEL.ParCELPosNegLP"\n')
        k = ",".join([ '"{}"'.format(x) for x in p if x.isascii() ])
        f.write(f'lp.positiveExamples = {{ {k} }}\n')
        k = ",".join([ '"{}"'.format(x) for x in n if x.isascii() ])
        # k = ",".join(map(lambda x : f'"{x}"',n))
        f.write(f'lp.negativeExamples = {{ {k} }}\n')
        f.write('algorithm.type = "org.dllearner.algorithms.ParCELEx.ParCELearnerExV2"\n')
        f.write('algorithm.maxExecutionTimeInSeconds = 60\n')
        f.write("algorithm.numberOfWorkers = 1\n")
        f.write("algorithm.splitter = splitter\n")
        f.write('splitter.type = "org.dllearner.algorithms.ParCEL.split.ParCELDoubleSplitterV1"\n')        
        # f.write('alg.writeSearchTree = true\n')

def run_sparcel(kb_pth, ex_path, celoe_path):
    P,N = read_examples_from_json(ex_path)
    instance_to_sparcel(kb_pth, P, N, ".", "sparcel_instance")
    outpt = subprocess.check_output(["java", "-jar", SPARCEL_PATH, "./sparcel_instance.conf"])
    output = str(outpt)
    lines = output.split("\\n")

    query = lines[-7]

#    query_red = query[8:]        
    print(query)

    print(lines[-5:-1])

    tp = int(lines[-5].split(":")[1])
    fp = int(lines[-4].split(":")[1])
    tn = int(lines[-3].split(":")[1])
    fn = int(lines[-2].split(":")[1])

    return (tp + tn) / (tp + fp + tn + fn), query

def run_celoe(kb_path, ex_path):
    P,N = read_examples_from_json(ex_path)
    confpath = os.path.join(os.path.dirname(ex_path),"dllearner_instance.conf")    
    instance_to_dllearner(kb_path, P, N, os.path.dirname(ex_path), "dllearner_instance")    
    outpt = subprocess.check_output([CELOE_PATH, confpath])
    output = str(outpt)
    lines = output.split("\\n")
    i = 0
    for l in lines:
        if l == "solutions:":
            break
        i += 1
    return float(lines[i+1][lines[i+1].find("pred. acc.:")+12:lines[i+1].find("%, F")])
    
def solve_fixed_k(path, ex_path, k):
    A = structure_from_owl(path)
    P,N = read_examples_from_json(ex_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND,NEG})
    return f.solve()


def solve(path, ex_path, k):
    A = structure_from_owl(path)
    P,N = read_examples_from_json(ex_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND,NEG})
    return f.solve_incr(k, return_string= True)

def solve_old(path, ex_path, k):
    A = structure_from_owl(path)
    P,N = read_examples_from_json(ex_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = fitting_alc1.FittingALC(A,k,P,N, op = {EX,ALL,OR,AND, NEG})
    return f.solve()
    
def test(path,P,N):
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,6,P,N, op = {EX,ALL,OR,AND})
    f.solve()

def run_on_ontolearn_examples(kb_path, json_path, problem_key, k):
    A = structure_from_owl(kb_path)    
    with open(json_path) as f:
        d = json.load(f)
    P = d["problems"][problem_key]["positive_examples"]
    N = d["problems"][problem_key]["negative_examples"]
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND})    
    f.solve_incr(k)

def instance_to_dllearner(kb_path, p, n, dest, file_name = "dl_instance"):
    file = os.path.join(dest,f"{file_name}.conf")
    with open(file, "w+") as f:
        f.write('ks.type = "OWL File"\n')
        f.write(f'ks.fileName = "{kb_path}"\n')
        f.write('measure.type = "gen_fmeasure"')
        f.write('reasoner.type = "closed world reasoner"\n')
        f.write('reasoner.sources = { ks }\n')
        f.write('lp.type = "posNegStandard"\n')
        k = "{" + ",".join(map(lambda x : f'"{x}"',p)) + "}"
        f.write(f'lp.positiveExamples = {k}\n')
        k = "{" +  ",".join(map(lambda x : f'"{x}"',n)) + "}"
        f.write(f'lp.negativeExamples = {k}\n')
        f.write('alg.type = "celoe"\n')
        f.write('alg.maxExecutionTimeInSeconds = 300\n')
        f.write('alg.writeSearchTree = false\n')        
        f.write('h.type ="celoe_heuristic"\n')
        f.write('h.expansionPenaltyFactor = 0.02\n')
        f.write('alg.stopOnFirstDefinition = true\n')        
#        f.write('alg.maxNrOfResults = 1\n')
        #f.write('useMinimizer = false\n')
        #alg.noisePercentage = 32                
        #//alg.maxClassDescriptionTests = 10000000

def json_to_dllearner(kb_path, json_path, dest_dir):
    with open(json_path) as f:
        d = json.load(f)
        instance_to_dllearner(kb_path, d["P"] ,d["N"], dest_dir, Path(json_path).stem)

def jsons_to_dllearner(kb_path, dir, dest_dir):
    for file in os.listdir(dir):
        if os.path.splitext(file)[1] == '.json':
            json_to_dllearner(kb_path, os.path.join(dir, file), dest_dir)

def reduce_size_by_examples(kb_path, json_path, newpath, filename, k):
    P,N = read_examples_from_json(json_path)
    A = structure_from_owl(kb_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    B,m = restrict_to_neighborhood(k-1, A, P + N)
    construct_owl_from_structure(os.path.join(newpath,f"{filename}.owl"),B)

def examples_by_queries(kb_path, queries_path, q_pos, q_neg, n_pos, n_neg, dest_dir, file_name , random_pos = True, random_neg = True, exclude_pos_from_neg = False):
    g = get_ontology(kb_path).load()   
    graph = default_world.as_rdflib_graph() 
    d = dict()    
    with open(queries_path, 'r') as f:
        dq = json.load(f)
    d["q_pos"] = dq[q_pos]
    if q_neg is not None:
        d["q_neg"] = dq[q_neg]
    else:
        d["q_neg"] = "complement"    
    p_res = list( map(lambda x : x[0].get_iri(), default_world.sparql(dq[q_pos]["SPARQL"])))
    #p_res = list(map(lambda x : x[0].get_iri(),list(  graph.query_owlready(dq[q_pos]["SPARQL"]))))
    print(f"Positive:{len(p_res)}")
    if not p_res or n_neg>len(p_res):
        return False
    if random_pos:
        P = random.sample(p_res,n_pos)
    else:
        P = p_res[:n_pos]            

    if q_neg is not None:        
        #n_res_r = list( map(lambda x : x[0].get_iri(),list(graph.query_owlready(dq[q_neg]["SPARQL"]))))
        n_res_r = list( map(lambda x : x[0].get_iri(), default_world.sparql(dq[q_neg]["SPARQL"])))
    else:
        n_res_r = list(map(lambda x : x[0].get_iri(), default_world.sparql(QALL)))
    n_res = []
    if q_neg is None or exclude_pos_from_neg:
        for e in n_res_r:
            if not e in p_res:
                n_res.append(e)
    else:
        n_res = n_res_r
    print(f"Negative:{len(n_res)}")
    if not n_res or n_neg>len(n_res):
        return False
    if random_neg:
        N = random.sample(n_res,n_neg)
    else:
        N = n_res[:n_pos]
    d["n_pos"] = len(P)
    d["n_neg"] = len(N)
    d["P"] = P
    d["N"] = N
    d["rnd_pos"] = random_pos
    d["rnd_neg"] = random_neg
    d["random_seed"] = RANDOM_SEED
    with open(os.path.join(dest_dir,file_name), "w+") as f:
        json.dump(d,f)
    return True

def benchmark_run(dir):
    cols = ["data set","t_celoe", "t_evo", "t_spacel", "t_alcsat", "a_celoe","a_evo", "a_spacel", "a_alcsat"]    
    data = []
    js_path = None
    kb_path = None
    dsname = None
    for d in filter(lambda x: not x.startswith('.'),os.listdir(dir)):        
        dataset_dir = os.path.join(dir,d)
        if os.path.isdir(dataset_dir):
            for f in filter(lambda x: not x.startswith('.'),os.listdir(dataset_dir)):
                path = os.path.join(dataset_dir,f)
                base,ext = os.path.splitext(path)
                if ext == ".json":
                    js_path = path
                if ext == ".owl":
                    kb_path = path
                dsname = os.path.basename(dataset_dir)
            print(f"Running on {dsname}")
            P,N = read_examples_from_json(js_path)
            A = structure_from_owl(kb_path)
            
            start = time.time()
            a_celoe = run_celoe(kb_path,js_path)
            end = time.time()
            t_celoe = end-start

            start = time.time()
            a_evo, c_evo = run_evo(kb_path,P,N)
            end = time.time()
            t_evo = end-start

            start = time.time()
            a_sparcel, c_sparcel = run_sparcel(kb_path, js_path)
            end = time.time()
            t_sparcel = end-start

            P = list(map(lambda n: map_ind_name(A, n), P))
            N = list(map(lambda n: map_ind_name(A, n), N))       
            start = time.time()
            max_k = 32
            f = FittingALC(A,max_k,P,N, op = {EX,ALL,OR,AND,NEG})
            a_alcsat, n_alcsat,c_alcsat = f.solve_incr(max_k)
            end = time.time()
            t_alcsat = end-start

            data.append([dsname,t_celoe,t_evo,t_sparcel,t_alcsat,a_celoe, a_evo, a_sparcel, a_alcsat])            
            pd.DataFrame(data, columns=cols).to_csv(os.path.join(dir,'results_reproduced.csv'))

def benchmark_gen(kb_path, queries_path, dest_dir, q_ind, n_pos, n_neg, complement_for_neg = False):    
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)        
    red_kb_path_filename = f"yago_family_reduced_kb_{q_ind}_({n_pos},{n_neg})"
    js_path = os.path.join(dest_dir,f"{red_kb_path_filename}.json")
    q_pos = f"Q_p{q_ind}"
    if not complement_for_neg:
        q_neg = f"Q_n{q_ind}"
    else:
        q_neg = None
    if not os.path.exists(js_path):        
        if not examples_by_queries(kb_path, queries_path, q_pos,q_neg, n_pos, n_neg, dest_dir, js_path ):
            return
    reduce_size_by_examples(kb_path,js_path,dest_dir,red_kb_path_filename,5)

def kCrossVal(P,N,k):
    def toPN(Sp):
        Pp = list(filter(lambda x : x[1] == 1, Sp))
        Nn = list(filter(lambda x : x[1] == 0, Sp))
        return Pp,Nn
    n = len(P) + len(N)
    S = [(x,0) for x in N] + [(x,1) for x in P]
    random.shuffle(S)
    subsamples = []
    i = 0
    for i in range(k-1):
        subsamples.append(S[i:i+(n//k)])
        i += n//k
    subsamples.append(S[i:])
    for i in range(k):
        yield (toPN(sum(subsamples[0:i]) + sum(subsamples[i+1:k+1])),toPN(subsamples[i]))

def to_tex(path):
    df = pd.read_csv(path)
    df.style.format(decimal=',', thousands='.', precision=2).to_latex(os.path.join(os.path.dirname(path),"tex.txt"))

# 1: f"yago_family_m_and_f_{k}-descendant_p{i}-n{i}"
# 2: f"yago_family_and_paths_d3_8-{k}"

def benchmark_gen_t():    
    if not os.path.exists(sys.argv[3]):
        os.mkdir(sys.argv[3])
    i = 200
    for k in [6]:#range(4,5):
        dest_dir = os.path.join(sys.argv[3], f"yago_language_{k}_{i}")
        benchmark_gen(sys.argv[1], sys.argv[2], dest_dir, k,i,i)

def convertToTikzCsv(files):
    compare_cols = ['t_celoe','t_evo','t_spacel']
    result_csv = ["family_celoe_alcsat_time.csv", "family_evo_alcsat_time.csv", "family_sparcel_alcsat_time.csv"]
    
    nd = [None,None,None]
    for f in files:
        d = pd.read_csv(f)
        for i,c in enumerate(compare_cols):
            if nd[i] is None:
                nd[i] = d[[c, 't_alcsat']]
            else:
                nd[i] = pd.concat([nd[i],d[[c, 't_alcsat']]])
    for i in range(3):
        nd[i].to_csv(result_csv[i], index=False )

def convertToTikzCsvTwoFiles(files):
    compare_col = 't_alcsat'#'t_evo' 't_sparcel'
    result_csv = "family_alcsat_alcsat+_time.csv" #"family_celoe_alcsat_time.csv" #"family_evo_alcsat_time.csv" "family_sparcel_alcsat_time.csv"
    
    if len(files) % 2==0:
        nd = None
        i = 0
        while i < len(files):
            d1 = pd.read_csv(files[i])
            d2 = pd.read_csv(files[i+1]).rename(columns = {'t_alcsat' : 't_alcsat+'})            
            d = d1[['t_alcsat']]
            d = pd.concat([d,d2[['t_alcsat+']]], axis = 1)
            if nd is None:
                nd = d
            else:                
                nd = pd.concat([nd,d])
            i += 2
        nd.to_csv(result_csv, index=False)

def convertCsv(files):

    compare_cols = ['t_celoe','t_evo','t_spacel']
    result_csv = ["family_celoe_alcsat_time.csv", "family_evo_alcsat_time.csv", "family_sparcel_alcsat_time.csv"]

    time_intervals = range(0,241,20) 

    data = pd.DataFrame(index=time_intervals,columns=['n_alcsat', 'n_evo', 'n_sparcel', 'n_celoe'])

    nd = None
    for f in files:
        d = pd.read_csv(f)        
        if nd is None:
            nd = d
        else:
            nd = pd.concat([nd,d])    
    for t in time_intervals:
        n_alcsat = 0
        n_celoe = 0
        n_sparcel = 0
        n_evo = 0
        for i in range(nd.shape[0]):
            if nd.iloc[i]['t_alcsat'] <= t and nd.iloc[i]['a_alcsat'] == 1.0:
                n_alcsat +=1
            if nd.iloc[i]['t_celoe'] <= t and nd.iloc[i]['a_celoe'] == 100.0:
                n_celoe +=1
            if nd.iloc[i]['t_spacel'] <= t and nd.iloc[i]['a_spacel'] == 1.0:
                n_sparcel +=1
            if nd.iloc[i]['t_evo'] <= t and nd.iloc[i]['a_evo'] == 1.0:
                n_evo +=1
        data.at[t,'n_alcsat'] = n_alcsat
        data.at[t,'n_celoe'] = n_celoe
        data.at[t,'n_sparcel'] = n_sparcel
        data.at[t,'n_evo'] = n_evo
    data.to_csv("data_graph.csv")

def main():    
    benchmark_run(sys.argv[1])

if __name__ == "__main__":
    main()