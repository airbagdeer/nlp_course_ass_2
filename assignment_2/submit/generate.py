from collections import defaultdict
import random

class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen(self, symbol, show_tree=False):
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion(symbol)
            gen_expansion = " ".join(self.gen(s, show_tree=show_tree) for s in expansion)
            if show_tree:
                return f"({symbol} {gen_expansion})"
            return gen_expansion

    def random_sent(self, show_tree=False):
        return self.gen("ROOT", show_tree=show_tree)

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("grammar_file", type=str, help="Path to the grammar file")
    parser.add_argument("-n", type=int, required=False, default=1)
    parser.add_argument("-t", action="store_true", help="Output tree structure")

    args=parser.parse_args()
    n=args.n
    # pcfg = PCFG.from_file("/Users/eyal/Documents/nlp_course/assignment_2/grammar")
    pcfg = PCFG.from_file(args.grammar_file)

    for i in range(n):
        print(pcfg.random_sent(show_tree=args.t))
