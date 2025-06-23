import sys
import time
import argparse
from spell.fitting import solve_incr, mode 
from spell.fitting_alc import *

from spell.structures import solution2sparql, structure_from_owl

LANGUAGES = ["el", "el_alcsat", "fl0", "ex-or", "all-or", "elu", "alc"]
L_OP = {
    "el" : [EX,AND],
    "el_alcsat" : [EX,AND],
    "fl0" : [ALL, AND],
    "ex-or" : [EX, OR],
    "all-or" : [ALL,OR],
    "elu" : [EX,OR,AND],
    "alc" : [EX,OR,AND,OR,NEG]
}

def main():
    parser = argparse.ArgumentParser(prog="SPELL")

    parser.add_argument(
        "kb_owl_file", help="path to a OWL knowledge base in RDF/XML format"
    )
    parser.add_argument(
        "pos_example_list", help="path to a textfile containing positive examples"
    )
    parser.add_argument(
        "neg_example_list", help="path to a textfile containing negative examples"
    )

    parser.add_argument("--language", type=str, default="el",choices=LANGUAGES, help = "language to learn in, el: {exists,and}, el_alcsat: {exists,and}, fl0: {forall,and}, ex-or: {exists,or}, all-or: {forall,or}, elu: {exists,and,or}, alc: {forall,exists,and,or,neg} (default=el)")

    parser.add_argument("--max_size", type=int, default=12, help="(default=12)")
    parser.add_argument(
        "--mode",
        choices=["exact", "neg_approx", "full_approx"],
        default=mode.exact,
        help="(default=exact)",
    )

    parser.add_argument(
        "--output", type=str, help="write best fitting SPARQL query to a file"
    )
    parser.add_argument(
        "--timeout", type=float, default=-1, help="in seconds (default=-1)"
    )

    parser.add_argument("--disable_tree_templates", action='store_true', help ='(alcsat only) disables optimization that precomputes tree templates')
    parser.add_argument("--disable_type_encoding", action='store_true', help = '(alcsat only) disables optimization that replaces concept names with types (internally)')

    args = parser.parse_args()

    owlfile = args.kb_owl_file
    pospath = args.pos_example_list
    negpath = args.neg_example_list

    md = args.mode

    time_start = time.perf_counter()

    print("== Loading {}".format(owlfile))
    A = structure_from_owl(owlfile)

    P: list[int] = []
    with open(pospath, encoding="UTF-8") as file:
        for line in file.readlines():
            ind = line.rstrip()
            if ind not in A.indmap:
                print(
                    "[ERR] The positive example {} does not seem to occur in {}".format(
                        ind, owlfile
                    )
                )
                sys.exit(1)
            P.append(A.indmap[ind])

    N: list[int] = []
    with open(negpath, encoding="UTF-8") as file:
        for line in file.readlines():
            ind = line.rstrip()
            if ind not in A.indmap:
                print(
                    "[ERR] The negative example {} does not seem to occur in {}".format(
                        ind, owlfile
                    )
                )
                sys.exit(1)
            N.append(A.indmap[ind])

    time_parsed = time.perf_counter()

    print("== Starting incremental search search for fitting query")
    time_start_solve = time.perf_counter()

    if args.language in LANGUAGES - ["el"]:
        f = FittingALC(A, args.max_size, P, N, op = L_OP[args.language], type_encoding=not args.disable_type_encoding, tree_templates=not args.disable_tree_templates)
            
        remaining_time = -1
        if args.timeout != -1:
            remaining_time = args.timeout - (time.perf_counter() - time_start)
        if args.mode == "exact":
            f.solve_incr(args.max_size, timeout=remaining_time)
        elif args.mode == "full_approx":
            f.solve_incr_approx(args.max_size, timeout=remaining_time)
        else:
            print(f"Mode {args.mode} is only supported for SPELL.")
    else:
        _, res = solve_incr(A, P, N, md, timeout=args.timeout, max_size=args.max_size)

    time_solved = time.perf_counter()

    print(
        "== Took {:.2f}s for reading input and {:.3f}s for solving".format(
            time_parsed - time_start, time_solved - time_start_solve
        )
    )

    if args.output != None:
        print("== Writing result to {}".format(args.output))
        with open(args.output, "w", encoding="UTF-8") as file:
            file.write(solution2sparql(res))


if __name__ == "__main__":
    main()
