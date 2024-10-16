## Converting Words to Features

### One-Hot Encoding
* Converts categorical data (e.g., words) into numerical feature vectors.
* Each unique word has a corresponding vector with a single element set to 1 and the rest set to 0.
* The dimension of the vector is equal to the size of the vocabulary.

### Bag of Words
* Represents a document as the aggregate or average of one-hot encoded vectors for its words.
* Disregards the order of words and focuses on their frequency.

### Embeddings
* Maps words to dense, low-dimensional vectors.
* Captures semantic relationships between words.
* Often used in place of one-hot encoding due to their efficiency and ability to capture semantic information.

### Embedding Bags
* A layer in neural networks that efficiently computes the average or sum of embeddings for a sequence of words.
* Commonly used in NLP tasks like text classification and sentiment analysis.

### PyTorch Implementation
* Tokenization: Converts text into a sequence of token IDs.
* Embedding Layer: Maps token IDs to embedding vectors.
* Embedding Bag Layer: Computes the average or sum of embeddings for a sequence of words.
* Offset Parameter: Specifies the starting position of each document in a batch.

