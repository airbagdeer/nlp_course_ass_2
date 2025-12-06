import sys
from collections import defaultdict, Counter

def read_data(filename):
    """
    Reads data in the format: Word/Tag
    Returns a list of sentences, where each sentence is a list of (word, tag) tuples.
    """
    sentences = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # The file format seems to be space-separated Word/Tag tokens
            # Example: -DOCSTART-/O EU/I-ORG rejects/O ...
            tokens = line.split()
            sentence = []
            for token in tokens:
                # Handle cases where word might contain '/'
                # Split from the rightmost slash
                if '/' in token:
                    word, tag = token.rsplit('/', 1)
                else:
                    # Should not happen based on format, but safe fallback
                    word, tag = token, 'O'
                sentence.append((word, tag))
            sentences.append(sentence)
    return sentences

def train_counts(sentences):
    """
    Counts word-tag occurrences.
    Returns:
    - word_to_tag: mapping word -> most_frequent_tag
    - vocab: set of all words seen in training (for casing checks)
    """
    word_tag_counts = defaultdict(Counter)
    vocab = set()
    for sentence in sentences:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
            vocab.add(word)
    
    word_to_tag = {}
    for word, counts in word_tag_counts.items():
        # Get the tag with the highest count
        best_tag = counts.most_common(1)[0][0]
        word_to_tag[word] = best_tag
    
    return word_to_tag, vocab

def predict(sentences, word_to_tag, vocab):
    """
    Predicts tags for sentences using the word_to_tag map and heuristics.
    Returns a list of sentences, where each sentence is a list of (word, predicted_tag) tuples.
    """
    predictions = []
    for sentence in sentences:
        pred_sentence = []
        for i, (word, _) in enumerate(sentence):
            if word in word_to_tag:
                pred_tag = word_to_tag[word]
            else:
                # Heuristics for unknown words
                if word == "-DOCSTART-":
                    pred_tag = "O"
                # Capitalized words might be entities
                elif word and word[0].isupper():
                    # Check suffixes
                    if word.endswith("stan") or word.endswith("ia") or word.endswith("land"):
                        pred_tag = "I-LOC"
                    elif word.endswith("son") or word.endswith("ov") or word.endswith("ski"):
                        pred_tag = "I-PER"
                    elif word.endswith("Inc") or word.endswith("Ltd") or word.endswith("Corp") or \
                         word.endswith("Group") or word.endswith("Bank") or word.endswith("News") or \
                         word.endswith("University") or word.endswith("Association") or word.endswith("Party") or \
                         word.endswith("Council") or word.endswith("Committee"):
                        pred_tag = "I-ORG"
                    else:
                        # Default for capitalized words
                        if i == 0:
                            # Start of sentence:
                            # If the lowercase version is in vocab (e.g. "The", "In"), likely O.
                            # Otherwise, treat as proper noun (default I-PER).
                            if word.lower() in vocab:
                                pred_tag = "O"
                            else:
                                pred_tag = "I-PER"
                        else:
                            pred_tag = "I-PER" # Guess person for capitalized words mid-sentence
                else:
                    pred_tag = "O"
            
            pred_sentence.append((word, pred_tag))
        predictions.append(pred_sentence)
    return predictions

def write_preds(predictions, output_file):
    """
    Writes predictions to output file in the same format: Word/Tag
    """
    with open(output_file, 'w') as f:
        for sentence in predictions:
            line_tokens = [f"{word}/{tag}" for word, tag in sentence]
            f.write(" ".join(line_tokens) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ner.py <train_file> <test_file> [output_file]")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "NER_preds.txt"

    print(f"Reading training data from {train_file}...")
    train_sentences = read_data(train_file)
    
    print("Training counts...")
    word_to_tag, vocab = train_counts(train_sentences)
    
    print(f"Reading test data from {test_file}...")
    test_sentences = read_data(test_file)
    
    print("Predicting...")
    predictions = predict(test_sentences, word_to_tag, vocab)
    
    print(f"Writing predictions to {output_file}...")
    write_preds(predictions, output_file)
    print("Done.")
