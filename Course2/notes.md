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

## Summary of the Lecture Note on Language Modeling with N-Grams

### Key Concepts
* **N-grams:** Sequences of n words.
* **Bi-gram:** A sequence of two words.
* **Tri-gram:** A sequence of three words.
* **Language Modeling:** Predicting the next word in a sequence given the previous words.

### N-Gram Models
* **Conditional probability models:** Predict the probability of a word given its preceding context.
* **Context size:** The number of previous words considered for prediction.
* **Bi-gram models:** Use the immediate previous word as context.
* **Tri-gram models:** Use the two previous words as context.

### N-Gram Model Implementation
* **Vocabulary:** A set of unique words.
* **Context vector:** A numerical representation of the previous words.
* **Probability calculation:** Using conditional probability tables or neural networks.
* **Prediction:** Choosing the word with the highest probability.

### Neural Network Approach
* **Context vector:** A concatenation of embedding vectors for the previous words.
* **Neural network architecture:** A feedforward neural network with input, hidden, and output layers.
* **Prediction:** The output layer predicts the next word.

### Key Points
* N-gram models are a simple but effective approach to language modeling.
* Larger n-grams can capture more context but require more data.
* Neural networks offer a more powerful and flexible approach to language modeling.
* The choice of n-gram model or neural network depends on the specific task and available data.

## Summary of the Lecture Note on N-Grams as Neural Networks with PyTorch

### Key Points
* **N-gram model:** A language model that predicts the next word in a sequence based on a fixed number of previous words.
* **PyTorch implementation:** A classification model that uses a context vector as input.
* **Sliding window:** A technique to extract context and target words from a sequence.
* **Training:** Prioritize loss over accuracy as a performance metric.
* **Prediction:** Generate a sequence of words using the trained model.

### Steps Involved
1. **Create an embedding layer:** Specify vocabulary size, embedding dimension, and context size.
2. **Create the context vector:** Concatenate embedding vectors for the context words.
3. **Build the neural network:** Use a classification model with an extra hidden layer.
4. **Create a sliding window:** Extract context and target words from the sequence.
5. **Train the model:** Minimize the loss function.
6. **Predict:** Input a context and get the predicted next word.

### Key Concepts
* **Context vector:** A numerical representation of the previous words.
* **Sliding window:** A technique to extract context and target words.
* **Loss function:** A metric to measure the model's performance.
* **Prediction:** The process of generating the next word based on the context.


