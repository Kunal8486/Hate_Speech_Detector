# Logistic Regression

Converts text into numerical feature vectors based on word importance.

TF-IDF (Term Frequency – Inverse Document Frequency) increases the weight of rare, informative words and reduces weight for very common words like “the”, “is”, “to”.

TF-IDF helps Logistic Regression because it creates sparse, linear-friendly feature vectors.
Bigrams help capture short phrases (like “go back” or “you people”) crucial in hate speech detection.

Including bigrams improves context understanding, boosting recall slightly.

But still, TF-IDF cannot capture semantic meaning (like sarcasm or word order beyond 2-grams) → limits F1 score ceiling to around 0.75.

Need more context to recognize hate
Pros: Simple, interpretable, fast, consistent.
Cons: Linear → can’t capture complex patterns or multi-word context.

# Naive Bayes

Probabilistic classifier assuming words occur independently given the label.
Uses word frequency patterns (multinomial distribution).
Pros: Simple, efficient, works well on sparse TF-IDF.
Cons: Fails to model dependencies between words (e.g., “not racist” → misclassified).
Why it’s worse: The independence assumption makes it oversensitive to individual hate terms, ignoring context.

# Random Forest

Pros: Handles non-linear relations, robust, interprets complex patterns (e.g., “go back to your country”).
Cons: Slower, higher memory use, may overfit if not tuned.
Why it’s best: Captures interaction effects between words that linear models can’t (e.g., “you” + “people” = hate context only together).

# Support Vector Machine

Finds the maximum-margin hyperplane separating hate and non-hate.

Uses linear kernel (like Logistic Regression) but focuses on support vectors — the most informative samples near the decision boundary.
Pros: Works well on high-dimensional sparse TF-IDF data, robust to noise.
Cons: Still linear, doesn’t capture multiword nonlinearity like Random Forest.

Why it performs between LR and RF:
It’s linear like Logistic Regression but margin-based optimization gives it better generalization.

# KNN

K-Nearest Neighbors finds the k (5 in this case) closest samples (neighbors) to a given input using a distance metric, and classifies it based on the majority vote (or weighted vote).

weights='distance' → Closer neighbors have higher influence on classification.

metric='cosine' → Measures text similarity based on angle between vectors, which works well for TF-IDF.

fit() trains (stores) the data points in memory for distance comparisons later.