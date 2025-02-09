from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LFUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}  # Maps index to (embedding, response)
        self.frequency = defaultdict(int)  # Maps index to frequency count
        self.order = OrderedDict()  # Maps index to insertion order based on frequency

    def add(self, embedding, response):
        if len(self.cache) >= self.max_size:
            self.evict()

        # Add new entry
        index = len(self.cache)
        self.cache[index] = (embedding, response)
        self.frequency[index] = 1
        self.order[index] = self.frequency[index]

    def evict(self):
        # Find the least frequently used item
        lfu_idx = min(self.order, key=lambda k: self.order[k])

        # Remove from cache and frequency tracker
        del self.cache[lfu_idx]
        del self.frequency[lfu_idx]
        del self.order[lfu_idx]

    def find_best_match(self, query_embedding):
        if not self.cache:
            return None

        # Convert query_embedding to a numpy array if it's not already
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Compute cosine similarity
        embeddings = np.vstack([emb for emb, _ in self.cache.values()])
        similarities = cosine_similarity(query_embedding, embeddings)

        # Find the index of the maximum similarity
        best_match_idx = np.argmax(similarities)
        best_score = similarities[0, best_match_idx]

        if best_score > 0.8:
            idx = list(self.cache.keys())[best_match_idx]
            # Update frequency and order
            self.frequency[idx] += 1
            self.order[idx] = self.frequency[idx]
            return self.cache[idx][1]
        else:
            return None

    def print_cache(self):
        print("Cache Contents:")
        for idx, (embedding, response) in self.cache.items():
            print(f"Index: {idx}, Response: {response}, Frequency: {self.frequency[idx]}")
        print("------------------------------------------\n")