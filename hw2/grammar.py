"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        result = True
        for key in self.lhs_to_rules.keys():
            if not key.isupper():
                result = False
            #print('testhere',[x[1] for x in self.lhs_to_rules[key]])
            for item in [x[1] for x in self.lhs_to_rules[key]]:
                if len(item) == 2 and item[0].isupper() and item[1].isupper():
                    pass
                elif len(item) == 1:
                    for j in item[0]:
                        if j.isupper():
                            result = False
            prob_list = [x[-1] for x in self.lhs_to_rules[key]]
            if abs(fsum(prob_list))< 0.00001:
                #print(prob_list,fsum(prob_list))
                result = False
        return result


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
    testresult = grammar.verify_grammar()
    if testresult == False:
        print("ERROR. It's not a valid PCFG in CNF")
    else:
        print("It's  a valid PCFG in CNF")


    #test code
    # ta = 'WHAT'
    # tb = 'RESTRICTIONS'
    # aaa = grammar.lhs_to_rules['FLIGHT']
    # print(aaa[0][1][0])
    # for i in range(2,5):
    #     print(i)
    #
    # print(max(1,201))
    # td = defaultdict(dict)
    # td[1]={}
    # td[1][0] = 1
    # print((("NP",0,2),("FLIGHTS",2,3))+(("NP",0,2),("FLIGHTS",2,3)))
    # print(get_tree(table, 0, len(toks), grammar.startsymbol))