import sys
import torch
import numpy as np
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using device: {device}")

# Load Model and Tokenizer
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

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
            tokens = line.split()
            sentence = []
            for token in tokens:
                if '/' in token:
                    word, tag = token.rsplit('/', 1)
                else:
                    word, tag = token, 'O'
                sentence.append((word, tag))
            sentences.append(sentence)
    return sentences

def get_sentence_embeddings(sentence_words):
    """
    Runs RoBERTa on a sentence and returns a list of embeddings (one per word).
    We use the embedding of the first subword token for each word.
    """
    # Tokenize and keep track of word alignment
    # We can use tokenizer(..., is_split_into_words=True)
    inputs = tokenizer(sentence_words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Last hidden state: [batch_size, seq_len, hidden_dim]
    last_hidden_state = outputs.last_hidden_state[0] # [seq_len, 768]
    
    word_ids = inputs.word_ids()
    
    word_embeddings = []
    current_word_idx = None
    
    # word_ids maps tokens to original word indices. None for special tokens.
    # We take the first token for each word.
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != current_word_idx:
            word_embeddings.append(last_hidden_state[idx])
            current_word_idx = word_idx
            
    # Safety check: ensure we have one embedding per word
    if len(word_embeddings) != len(sentence_words):
        # This might happen if truncation occurred or something weird.
        # Fallback: pad or truncate
        if len(word_embeddings) < len(sentence_words):
            # Pad with zeros
            diff = len(sentence_words) - len(word_embeddings)
            for _ in range(diff):
                word_embeddings.append(torch.zeros(768).to(device))
        else:
            word_embeddings = word_embeddings[:len(sentence_words)]
            
    return word_embeddings

def train_model(sentences):
    """
    1. Builds word_to_tag map (Counts Baseline).
    2. Builds Memory Bank of embeddings for Entities (Tag != O).
    """
    word_tag_counts = defaultdict(Counter)
    memory_bank_embeddings = []
    memory_bank_tags = []
    
    print("Building Memory Bank...")
    for sentence in tqdm(sentences):
        words = [w for w, t in sentence]
        tags = [t for w, t in sentence]
        
        # Update counts
        for w, t in sentence:
            word_tag_counts[w][t] += 1
            
        # Get embeddings
        embeddings = get_sentence_embeddings(words)
        
        # Store entity embeddings
        for i, tag in enumerate(tags):
            if tag != 'O':
                memory_bank_embeddings.append(embeddings[i])
                memory_bank_tags.append(tag)
            elif np.random.rand() < 0.1: # Sample 10% of O tags
                memory_bank_embeddings.append(embeddings[i])
                memory_bank_tags.append(tag)
                
    # Counts Map
    word_to_tag = {}
    for word, counts in word_tag_counts.items():
        best_tag = counts.most_common(1)[0][0]
        word_to_tag[word] = best_tag
        
    # Stack memory bank
    if memory_bank_embeddings:
        memory_bank_tensor = torch.stack(memory_bank_embeddings) # [N, 768]
    else:
        memory_bank_tensor = torch.empty(0, 768).to(device)
        
    return word_to_tag, memory_bank_tensor, memory_bank_tags

def predict(sentences, word_to_tag, memory_bank_tensor, memory_bank_tags):
    """
    Predicts tags.
    - Known word -> Counts
    - Unknown word -> Nearest Neighbor in Memory Bank
    """
    predictions = []
    
    print("Predicting...")
    for sentence in tqdm(sentences):
        words = [w for w, t in sentence]
        pred_sentence = []
        
        # We only need embeddings if there are unknown words
        # But to keep logic simple, let's get embeddings for the whole sentence if needed
        # Optimization: Check if any word is unknown first?
        # Actually, getting embeddings is expensive. Let's do it only if needed.
        
        unknown_indices = [i for i, w in enumerate(words) if w not in word_to_tag]
        
        sentence_embeddings = None
        if unknown_indices and memory_bank_tensor.size(0) > 0:
            sentence_embeddings = get_sentence_embeddings(words)
            
        for i, word in enumerate(words):
            if word in word_to_tag:
                pred_tag = word_to_tag[word]
            else:
                # Unknown word
                if sentence_embeddings is not None:
                    emb = sentence_embeddings[i].unsqueeze(0) # [1, 768]
                    
                    # Cosine Similarity
                    # memory_bank: [N, 768]
                    # We want cosine sim.
                    # Normalize both
                    emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                    bank_norm = torch.nn.functional.normalize(memory_bank_tensor, p=2, dim=1)
                    
                    # Dot product
                    sims = torch.mm(emb_norm, bank_norm.t()) # [1, N]
                    
                    # Get best match
                    best_val, best_idx = torch.max(sims, dim=1)
                    
                    # Threshold?
                    if best_val.item() > 0.8: # Heuristic threshold
                        pred_tag = memory_bank_tags[best_idx.item()]
                    else:
                        pred_tag = 'O'
                else:
                    pred_tag = 'O'
            
            pred_sentence.append((word, pred_tag))
        predictions.append(pred_sentence)
    return predictions

def write_preds(predictions, output_file):
    with open(output_file, 'w') as f:
        for sentence in predictions:
            line_tokens = [f"{word}/{tag}" for word, tag in sentence]
            f.write(" ".join(line_tokens) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ner_roberta.py <train_file> <test_file> [output_file]")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "NER_preds.txt"

    print(f"Reading training data from {train_file}...")
    train_sentences = read_data(train_file)
    
    # Train
    word_to_tag, memory_bank_tensor, memory_bank_tags = train_model(train_sentences)
    print(f"Memory Bank Size: {len(memory_bank_tags)}")
    
    print(f"Reading test data from {test_file}...")
    test_sentences = read_data(test_file)
    
    # Predict
    predictions = predict(test_sentences, word_to_tag, memory_bank_tensor, memory_bank_tags)
    
    print(f"Writing predictions to {output_file}...")
    write_preds(predictions, output_file)
    print("Done.")
