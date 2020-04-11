"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        #initialization
        tdic = defaultdict(list)
        for i in range(len(tokens)):
            j = tokens[i]
            if tdic == []:
                return False
            tdic[i,i+1] = [x[0] for x in self.grammar.rhs_to_rules[j,]]
            # print('123',tdic[i, i + 1])
        #main loop
        for length in range(2,len(tokens)+1):
            for i in range(len(tokens)+1-length):
                j = i + length
                for k in range(i+1,j):
                    # print(i,k,j,tdic[i, k],tdic[k, j])
                    for a in tdic[i, k]:
                        for b in tdic[k, j]:
                            tdic[i, j] = list(set(tdic[i, j]).union(set([x[0] for x in self.grammar.rhs_to_rules[(a,b)]])))
                            # tdic[i, j] = tdic[i, j] + [x[0] for x in grammar.rhs_to_rules[(a,b)]]


                    # print((tdic[i, k],tdic[k, j]))
                    # print([x[0] for x in grammar.rhs_to_rules[(tdic[i, k],tdic[k, j])]])
                    # tdic[i,j] = tdic[i,j] + [x[0] for x in grammar.rhs_to_rules[(tdic[i, k],tdic[k, j])]]
        return tdic[0,len(tokens)] == [self.grammar.startsymbol]

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """

        # if self.is_in_language(tokens) == False:
        #     print("the grammar can NOT parse this sentence")
        #     return 0
        # initialization
        table = defaultdict(dict)
        probs = defaultdict(dict)

        for i in range(len(tokens)):
            j = tokens[i]
            probs[(i, i + 1)] = defaultdict(int)
            table[(i, i + 1)] = defaultdict(str)
            for x in self.grammar.rhs_to_rules[j,]:
                # print('123',x)
                probs[(i, i + 1)][x[0]] = x[2]
                table[(i, i + 1)][x[0]] = x[1][0]
        # print('testhere',table)

        # main loop
        for length in range(2, len(tokens) + 1):
            for i in range(len(tokens) + 1 - length):
                j = i + length
                #print('i','j',i,j)
                probs[(i, j)] = defaultdict(int)
                for k in range(i + 1, j):
                    # print(i,k,j,table[(i, k)].keys(),table[(k, j)].keys())
                    for a in table[(i, k)].keys():
                        for b in table[(k, j)].keys():
                            for q in self.grammar.rhs_to_rules[(a,b)]:
                                # print(table)
                                # print(probs)
                                # print(q,i,k,j)
                                if probs[(i, j)][q[0]] < q[2] * probs[(i, k)][a] * probs[(k, j)][b]:
                                    probs[(i, j)][q[0]] = q[2]*probs[(i,k)][a]*probs[(k,j)][b]
                                    table[(i, j)][q[0]] = ((a,i,k),(b,k,j))
                                #print(probs[(i, j)][q[0]], q[2] * probs[(i, k)][a] * probs[(k, j)][b],i,k,j)
        for i in probs.keys():
            for j in probs[i].keys():
                if probs[i][j]>0:
                    probs[i][j] = math.log(probs[i][j])


        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    tree = (nt,chart[i,j][nt][0],chart[i,j][nt][1])
    print('tree',tree)
    def sub_tree(otree):
        #print('0',otree)

        if type(chart[otree[1],otree[2]][otree[0]]) != str:
            # print('1', chart[otree[1][1],otree[1][2]][otree[1][0]])
            # print('2',chart[otree[2][1],otree[2][2]][otree[2][0]])
            return (otree[0], sub_tree(chart[otree[1],otree[2]][otree[0]][0]),sub_tree(chart[otree[1],otree[2]][otree[0]][1]))
        else:
            return (otree[0], chart[otree[1],otree[2]][otree[0]])



    tree = (tree[0],sub_tree(tree[1]),sub_tree(tree[2]))

    return tree
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        # print(table)
        # print(probs)

        #rint(table[(3,6)])
        #print(probs[(0,1)].values())
        assert check_table_format(table)
        assert check_probs_format(probs)
        #print(table[0, len(toks)].keys())
        print(table[0, len(toks)][grammar.startsymbol])
        print(probs[0, len(toks)][grammar.startsymbol])
        # print('gettree',get_tree(table, 0, len(toks), grammar.startsymbol))
        # print(table[0,6][grammar.startsymbol])
