import argparse
import time
import sys
import threading
import _thread
from generate import PCFG


# ==========================================
# Part 1: Earley Parser Implementation
# ==========================================

class State:
    def __init__(self, lhs, rhs, dot, start, end):
        self.lhs = lhs
        self.rhs = tuple(rhs)  # Make tuple for hashing
        self.dot = dot
        self.start = start
        self.end = end

    def next_symbol(self):
        if self.dot < len(self.rhs):
            return self.rhs[self.dot]
        return None

    def is_complete(self):
        return self.dot >= len(self.rhs)

    def __eq__(self, other):
        return (self.lhs == other.lhs and self.rhs == other.rhs and
                self.dot == other.dot and self.start == other.start and self.end == other.end)

    def __hash__(self):
        return hash((self.lhs, self.rhs, self.dot, self.start, self.end))

    def __repr__(self):
        # Display for debugging: S -> NP . VP (0, 2)
        r = list(self.rhs)
        r.insert(self.dot, ".")
        return f"{self.lhs} -> {' '.join(r)} ({self.start}, {self.end})"


class EarleyParser:
    def __init__(self, pcfg):
        self.pcfg = pcfg

    def parse(self, sentence):
        # Clean up sentence (remove punctuation spacing issues if any, though we assume space separated)
        tokens = sentence.strip().split()
        if not tokens:
            return False

        # Chart is a list of sets of States.
        chart = [set() for _ in range(len(tokens) + 1)]

        # Initialize with ROOT rules starting at 0
        self._predict('ROOT', 0, chart)

        # Main Earley Loop
        for i in range(len(tokens) + 1):
            queue = list(chart[i])
            processed = set(chart[i])

            idx = 0
            while idx < len(queue):
                state = queue[idx]
                idx += 1

                if state.is_complete():
                    self._complete(state, chart, queue, processed)
                else:
                    sym = state.next_symbol()
                    if self.pcfg.is_terminal(sym):
                        self._scan(state, i, tokens, chart)
                    else:
                        self._predict(sym, i, chart, queue, processed)

        # Success condition: ROOT -> ... . (0, len(tokens)) exists in the last chart entry
        for state in chart[-1]:
            if state.lhs == 'ROOT' and state.is_complete() and state.start == 0:
                return True
        return False

    def _predict(self, symbol, index, chart, queue=None, processed=None):
        if symbol not in self.pcfg._rules:
            return

        for (rhs, weight) in self.pcfg._rules[symbol]:
            new_state = State(symbol, rhs, 0, index, index)
            self._add_to_chart(new_state, chart[index], queue, processed)

    def _scan(self, state, index, tokens, chart):
        if index >= len(tokens):
            return

        sym = state.next_symbol()
        if sym == tokens[index]:
            new_state = State(state.lhs, state.rhs, state.dot + 1, state.start, index + 1)
            if new_state not in chart[index + 1]:
                chart[index + 1].add(new_state)

    def _complete(self, completed_state, chart, queue, processed):
        source_states = chart[completed_state.start]
        for state in source_states:
            if not state.is_complete() and state.next_symbol() == completed_state.lhs:
                new_state = State(state.lhs, state.rhs, state.dot + 1, state.start, completed_state.end)
                self._add_to_chart(new_state, chart[completed_state.end], queue, processed)

    def _add_to_chart(self, state, state_set, queue, processed):
        if state not in state_set:
            state_set.add(state)
            if queue is not None:
                queue.append(state)
                processed.add(state)


# ==========================================
# Part 2: Generation Stress Test
# ==========================================

class TimeoutException(Exception):
    pass


def timeout_timer(seconds, stop_event):
    if not stop_event.wait(seconds):
        _thread.interrupt_main()


def run_generation_test(pcfg, num_sentences, time_limit):
    print(f"\n[TEST] Generation Stress Test ({num_sentences} sentences)...")
    print(f"Goal: Ensure no infinite recursion loops (timeout {time_limit}s).")

    success_count = 0
    timeouts = 0
    recursion_errors = 0

    for i in range(num_sentences):
        stop_event = threading.Event()
        t = threading.Thread(target=timeout_timer, args=(time_limit, stop_event))
        t.daemon = True
        t.start()

        try:
            try:
                pcfg.random_sent()
                success_count += 1
            except KeyboardInterrupt:
                timeouts += 1
            except RecursionError:
                recursion_errors += 1
            finally:
                stop_event.set()
        except Exception as e:
            print(f"Error: {e}")

    print(f"Result: {success_count}/{num_sentences} generated successfully.")
    if timeouts > 0:
        print(f"WARNING: {timeouts} sentences timed out.")
    if recursion_errors > 0:
        print(f"WARNING: {recursion_errors} RecursionErrors.")

    if success_count == num_sentences:
        print("PASSED generation test.")
    else:
        print("FAILED generation test.")


# ==========================================
# Part 3: Parsing Accuracy Test (Expanded)
# ==========================================

def run_parsing_test(pcfg):
    print("\n[TEST] Parsing Correctness Test...")
    print("Checking grammar against A.2 requirements and A.4 extensions (b: Questions, e: Agreement)")

    parser = EarleyParser(pcfg)

    test_suites = {
        "A.2 Core Requirements (Positive)": [
            # ===== Required 10 sentences from A.2 =====
            "Sally ate a sandwich .",
            "Sally and the president wanted and ate a sandwich .",
            "the president sighed .",
            "the president thought that a sandwich sighed .",
            "it perplexed the president that a sandwich ate Sally .",
            "the very very very perplexed president ate a sandwich .",
            "the president worked on every proposal on the desk .",
            "Sally is lazy .",
            "Sally is eating a sandwich .",
            "the president thought that Sally is a sandwich .",

            # ===== Additional tests for A.2 phenomena =====
            # Simple intransitives
            "Sally worked .",
            "the sandwich sighed .",
            "every president worked .",
            "a pickle sighed .",

            # Simple transitives
            "the president wanted a pickle .",
            "Sally ate every sandwich .",
            "a president kissed the floor .",
            "every sandwich perplexed Sally .",

            # Copula with adjectives
            "the sandwich is delicious .",
            "every president is lazy .",
            "a pickle is fine .",
            "the proposal is very perplexed .",

            # Copula with NP
            "Sally is a president .",
            "the sandwich is a pickle .",
            "every proposal is a sandwich .",

            # Progressive aspect (is/was eating)
            "the president is eating a sandwich .",
            "Sally is working .",
            "every president is eating every sandwich .",
            "a pickle is sighing .",

            # Prepositional phrases
            "the president worked on a desk .",
            "Sally ate a sandwich on the desk .",
            "the proposal on the desk sighed .",
            "every president on the floor worked .",
            "the president on every desk is lazy .",
            "Sally worked on the proposal on every desk .",
            "the sandwich on the floor on the desk sighed .",

            # Coordination - NP coordination
            "Sally and the president sighed .",
            "the sandwich and the pickle and the proposal worked .",
            "Sally and every president ate a sandwich .",

            # Coordination - VP coordination
            "Sally sighed and worked .",
            "the president ate a sandwich and sighed .",
            "Sally worked and sighed and ate a sandwich .",

            # Coordination - mixed
            "Sally and the president ate and wanted a sandwich .",
            "the president and Sally sighed and worked on the desk .",

            # Clausal complements (that-clauses)
            "the president thought that Sally worked .",
            "Sally thought that the sandwich is delicious .",
            "the president thought that Sally ate a sandwich .",
            "every president thought that a sandwich sighed .",
            "Sally thought that the president is eating a sandwich .",
            "the president thought that Sally and the sandwich worked .",

            # Extraposition (it ... that)
            "it perplexed Sally that the president sighed .",
            "it perplexed the sandwich that every president worked .",
            "it perplexed every president that Sally is lazy .",
            "it perplexed the president that the sandwich is delicious .",
            "it perplexed Sally that the president ate a sandwich .",

            # Recursive adjectives (very very ... Adj)
            "the very perplexed president sighed .",
            "the very very lazy Sally ate a sandwich .",
            "a very very very fine sandwich is delicious .",
            "the very very very very perplexed pickle sighed .",
            "Sally is very lazy .",
            "the sandwich is very very delicious .",
            "the very very very lazy president worked on the desk .",

            # Complex combinations
            "the very perplexed president on the desk sighed .",
            "Sally and the very lazy president ate a sandwich .",
            "the president thought that the very perplexed sandwich sighed .",
            "it perplexed the very lazy president that Sally worked .",
            "the very fine president and Sally sighed and worked .",
            "Sally ate a very delicious sandwich on the desk .",
            "the president on the desk thought that Sally is very lazy .",
            "every very perplexed president on every desk worked .",
        ],

        "A.2 Core Requirements (Negative - Should Fail)": [
            # ===== Subcategorization violations =====
            # Strictly intransitive verbs taking objects
            "Sally sighed a sandwich .",
            "every sandwich sighed the president .",

            # Strictly transitive verbs missing objects
            "Sally wanted .",
            "the president perplexed .",

            # Copula violations
            "Sally is .",  # Missing complement
            "the president is sighed .",  # Verb instead of Adj/NP

            # ===== Bad extraposition =====
            "it perplexed the president a sandwich sighed .",  # Missing 'that'
            "it perplexed a sandwich ate Sally .",  # Missing NP and 'that'
            "perplexed the president that Sally sighed .",  # Missing 'it'

            # ===== Missing determiners =====
            "president ate sandwich .",
            "Sally ate pickle .",
            "proposal sighed .",
            "sandwich is delicious .",

            # ===== Bad word order =====
            "president the ate .",
            "ate Sally sandwich a .",
            "the ate president sandwich a .",
            "sandwich a ate Sally .",
            "lazy is Sally .",
            "delicious is sandwich the .",

            # ===== Sentence fragments =====
            "the president .",
            "a sandwich .",
            "Sally and the president .",
            "on the desk .",
            "ate a sandwich .",
            "very perplexed .",

            # ===== Bad coordination =====
            "Sally and .",
            "and the president sighed .",
            "Sally and and the president sighed .",
            "Sally ate and .",

            # ===== Progressive with wrong verb =====
            "Sally is eat a sandwich .",  # Should be 'eating'
            "the president is sigh .",  # Should be 'sighing'

            # ===== Bad PP attachment =====
            "on the desk sighed .",  # PP without NP
            "the president on sighed .",  # PP missing NP

            # ===== Complex violations =====
            "the president thought that ate a sandwich .",  # Missing subject in embedded clause
            "it perplexed that Sally sighed .",  # Missing NP object
            "the very president ate a sandwich .",  # 'very' without adjective
        ],

        "A.4(b) Yes/No Questions (Positive)": [
            # ===== Basic did-questions =====
            "did Sally eat a sandwich ?",
            "did the president work ?",
            "did every sandwich perplex the president ?",
            "did Sally kiss the pickle ?",

            # ===== Basic will-questions =====
            "will Sally eat a sandwich ?",
            "will the president work on the proposal ?",
            "will every president sigh ?",
            "will the sandwich perplex Sally ?",

            # ===== Copula questions with adjectives =====
            "is Sally lazy ?",
            "is the president perplexed ?",
            "is every sandwich delicious ?",
            "is a pickle fine ?",

            # ===== Copula questions with NP =====
            "is the president a sandwich ?",
            "is Sally a president ?",
            "is every sandwich a pickle ?",
            "is the proposal a sandwich ?",

            # ===== Progressive questions =====
            "is Sally eating a sandwich ?",
            "is the president working ?",
            "is Sally eating every pickle ?",
            "is every president sighing ?",

            # ===== Questions with PPs =====
            "did the president work on the desk ?",
            "did Sally eat a sandwich on the desk ?",
            "will the president work on every proposal ?",
            "is the sandwich on the desk ?",

            # ===== Questions with coordination =====
            "did Sally and the president eat a sandwich ?",
            "will Sally and the president work ?",
            "are Sally and the president lazy ?",

            # ===== Questions with adjectives =====
            "did the very perplexed president sigh ?",
            "will the very lazy Sally work ?",
            "is the very delicious sandwich on the desk ?",

            # ===== Complex questions =====
            "did every very perplexed president on the desk work ?",
            "will Sally and the president eat every sandwich on the desk ?",
            "is the very lazy president on the floor eating a sandwich ?",
        ],

        "A.4(b) Yes/No Questions (Negative - Should Fail)": [
            # ===== Wrong verb form with auxiliary =====
            "did Sally ate a sandwich ?",  # Should be 'eat'
            "did the president worked ?",  # Should be 'work'
            "will Sally ate a sandwich ?",  # Should be 'eat'
            "will the president sighed ?",  # Should be 'sigh'

            # ===== Wrong form with copula =====
            "is Sally eat a sandwich ?",  # Should be 'eating' or remove 'is'
            "is the president sigh ?",  # Should be 'sighing'
            "is Sally worked ?",  # Should be 'working'

            # ===== Missing auxiliary =====
            "Sally eat a sandwich ?",
            "the president work on the desk ?",
            "Sally lazy ?",

            # ===== Bad word order =====
            "eating Sally is a sandwich ?",
            "lazy is Sally ?",
            "eat did Sally a sandwich ?",
            "work will the president ?",

            # ===== Auxiliary with wrong complement =====
            "did Sally eating a sandwich ?",  # Progressive with 'did'
            "will the president eating a sandwich ?",  # Progressive with 'will'

            # ===== Double auxiliary =====
            "did is Sally eat a sandwich ?",
            "will did the president work ?",
            "is did Sally lazy ?",

            # ===== Declarative order (not inverted) =====
            "Sally did eat a sandwich ?",
            "the president will work ?",
            "Sally is lazy ?",  # Technically could be yes/no with intonation, but grammar should require inversion

            # ===== Missing verb =====
            "did Sally a sandwich ?",
            "will the president ?",
            "is the president ?",

            # ===== Wrong auxiliary for verb type =====
            "did Sally is lazy ?",  # Copula needs 'is' inversion, not 'did'
            "will the president is a sandwich ?",
        ],

        "A.4(e) Singular/Plural Agreement (Positive)": [
            # ===== Singular Agreement =====
            "the president eats a sandwich .",
            "Sally wants a pickle .",
            "every sandwich is delicious .",
            "the chief of staff sighs .",
            "a proposal is fine .",
            "the floor is a sandwich .",
            "Sally thinks that the president is lazy .",

            # ===== Plural Agreement =====
            "the presidents eat a sandwich .",
            "the sandwiches are delicious .",
            "presidents want pickles .",
            "all presidents work .",
            "many proposals are fine .",
            "the chiefs of staff sigh .",
            "Sally and the president eat a sandwich .",
            "the president and the chief of staff are lazy .",
            "presidents think that sandwiches are delicious .",

            # ===== Mixed Agreement =====
            "the president eats sandwiches .",
            "presidents eat a sandwich .",
            "every president wants pickles .",
            "many presidents want a pickle .",
        ],

        "A.4(e) Singular/Plural Agreement (Negative - Should Fail)": [
            # ===== Singular Subject - Plural Verb =====
            "the president eat a sandwich .",
            "Sally want a pickle .",
            "every sandwich are delicious .",
            "the chief of staff sigh .",
            "a proposal are fine .",

            # ===== Plural Subject - Singular Verb =====
            "the presidents eats a sandwich .",
            "the sandwiches is delicious .",
            "presidents wants pickles .",
            "all presidents works .",
            "many proposals is fine .",
            "Sally and the president eats a sandwich .",
        ]
    }


    total_passed = 0
    total_tests = 0

    for suite_name, sentences in test_suites.items():
        print(f"\n--- {suite_name} ---")
        is_negative_suite = "Negative" in suite_name

        for s in sentences:
            total_tests += 1
            parsed = parser.parse(s)

            if is_negative_suite:
                # We expect Failure (False)
                if not parsed:
                    print(f"[PASS] Rejected: {s}")
                    total_passed += 1
                else:
                    print(f"[FAIL] Incorrectly parsed: {s}")
            else:
                # We expect Success (True)
                if parsed:
                    print(f"[PASS] Parsed: {s}")
                    total_passed += 1
                else:
                    print(f"[FAIL] Could not parse: {s}")

    print("\n" + "=" * 30)
    print(f"PARSING TEST SUMMARY: {total_passed}/{total_tests} Passed")
    print("=" * 30)

    if total_passed == total_tests:
        print("Result: EXCELLENT (All scenarios covered)")
    else:
        print("Result: ISSUES FOUND (Check the [FAIL] lines above)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grammar", help="Path to grammar file")
    parser.add_argument("-n", type=int, default=100, help="Number of sentences for generation stress test")
    args = parser.parse_args()

    print(f"Loading {args.grammar}...")
    try:
        pcfg = PCFG.from_file(args.grammar)
    except Exception as e:
        print(f"Failed to load grammar: {e}")
        sys.exit(1)

    # 1. Run Generation Check
    run_generation_test(pcfg, args.n, 1)

    # 2. Run Parsing Check
    run_parsing_test(pcfg)