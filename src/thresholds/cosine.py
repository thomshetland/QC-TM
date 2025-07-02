from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore",message=".*getrow.*deprecated.*")

class CosineSimilarity:
    def __init__(self, word_profile, feature_name):
        self.word_profile = word_profile
        self.feature_name = feature_name
    
    def check_cosine(self, word1, word2, features, word_profile):
        if word1 not in features or word2 not in features:
            return -float("inf")
        
        i, j = features[word1], features[word2]
        
        word1profile = word_profile.getrow(i)
        word2profile = word_profile.getrow(j)

        similarity = cosine_similarity(word1profile, word2profile)[0][0]

        return similarity