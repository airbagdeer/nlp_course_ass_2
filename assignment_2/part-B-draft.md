# Part B.2: Named Entity Recognition (RoBERTa)

## Approach
I implemented a **Nearest Neighbor** approach using **RoBERTa embeddings**. Since training a classifier was not allowed, I used the pre-trained `roberta-base` model to extract contextualized embeddings and performed similarity search against a memory bank of training examples.

### 1. Memory Bank Construction (Training)
- I passed all training sentences through `roberta-base`.
- For each named entity (Tag != 'O'), I stored its embedding (first subword token) and its tag in a "Memory Bank".
- To handle unknown non-entities, I also stored a random sample (10%) of 'O' tags in the Memory Bank.

### 2. Prediction
For each word in the test/dev set:
- **Known Words**: If the word was seen in the training data, I used the **Counts Baseline** (most frequent tag), as this is extremely reliable for known entities.
- **Unknown Words**:
    - I extracted the contextualized embedding of the word using `roberta-base`.
    - I calculated the **Cosine Similarity** between this embedding and all embeddings in the Memory Bank.
    - I assigned the tag of the nearest neighbor (highest similarity).
    - If the nearest neighbor was an 'O' sample, the prediction was 'O'.

## Results (Dev Set)
I evaluated this approach on the development set (`data-ner/dev`) using `ner_eval.py`.

*   **Accuracy**: 95.7%
*   **Precision**: 70.6%
*   **Recall**: 81.2%
*   **F1 Score**: 75.5%

This approach outperformed the heuristic baseline (F1 73.8%), demonstrating the value of contextualized embeddings for handling unknown words. The inclusion of 'O' samples in the memory bank was crucial for improving precision.

### Breakdown by Type
*   **PER**: Excellent recall (89.4%) and good precision (81.6%).
*   **LOC**: Strong performance (P: 77.8%, R: 84.5%).
*   **ORG**: Hardest category (P: 55.8%, R: 71.1%).
*   **MISC**: Reasonable performance (P: 60.3%, R: 72.9%).
