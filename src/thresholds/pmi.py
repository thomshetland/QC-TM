class PMI:
    def __init__(self, word_profile, feature_name):
        self.word_profile = word_profile
        self.feature_name = feature_name
    
    def check_pmi(word1, word2, features, word_profile):
        if word1 not in features or word2 not in features:
                return -float("inf")
        
        i, j = features[word1], features[word2]
        pmi_value = word_profile[i, j]
        return pmi_value
    
