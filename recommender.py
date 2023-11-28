import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from ast import literal_eval

class Recommender:
    def __init__(self, ids):
        self.n_users = len(ids)
        self.ids = ids
        self.users = {userId:User() for userId in ids}
        self.tags = pd.read_csv('tags.csv')
        self.tags.vector = self.tags.vector.apply(literal_eval)
        self.seen = []
        self.liked = []
        self.disliked = []
        self.all = []
        
    def calcSimilarity(self):
        liked_mean = np.array(self.tags.loc[self.liked]['vector'].tolist()).mean(axis=0)
        similarities = cosine_similarity([liked_mean], np.array(self.tags['vector'].tolist()))[0]
        similarities = pd.DataFrame(similarities, index = tags.index)
        return similarities

    def recommend(self):
        similarities = self.calcSimilarity()
        recommended = {user:self.users[user].recommend(similarities, i+1) for i,user in enumerate(self.users)}
        return recommended
    
    def likes(self, movieId, userId):
        self.liked += [movieId]
        self.users[userId].liked += [movieId]
        self.all += [movieId]
        self.users[userId].all += [movieId]
        
    def dislikes(self, movieId, userId):
        self.disliked += [movieId]
        self.users[userId].disliked += [movieId]
        self.all += [movieId]
        self.users[userId].all += [movieId]
        
    def seen(self, movieId, userId):
        self.seen += [movieId]
        self.users[userId].seen += [movieId]
        self.all += [movieId]
        self.users[userId].all += [movieId]
        
    def match(self):
        if self.n_users > 1:
            common = set(self.users[self.ids[0]].liked) & set(self.users[self.ids[1]].liked)
            for i in range(2, self.n_users):
                common = common & set(self.users[self.ids[i]].liked)
            return common
        
        return len(self.users[ids[0]].liked) > 0 
    
    
class User:
    def __init__(self):
        self.liked = []
        self.disliked = []
        self.seen = []
        self.all = []
    
    def recommend(self, similarities, n = 1):
        similarities = similarities.drop(index = self.all)
        return similarities.nlargest(n, 0).index.values[-1]
