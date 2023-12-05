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

        self.tags = pd.read_csv('fit.csv', index_col = 0)
        self.tags.vector = self.tags.vector.apply(literal_eval)
        self.seen = []
        self.liked = []
        self.disliked = []
        self.all = []
        
    def calcSimilarity(self):
        likedAndSeen = pd.concat((self.tags.loc[self.liked]['vector'], self.tags.loc[self.seen]['vector']))
        liked_mean = np.array(likedAndSeen.tolist()).mean(axis=0)
        liked_mean = np.concatenate((liked_mean, [1]))
        vectors = np.array(self.tags['vector'].tolist())
        vectors = np.append(vectors, np.array([self.tags.liked.values.tolist()]).T, axis = 1)
        similarities = cosine_similarity([liked_mean], vectors)[0]
        similarities = pd.DataFrame(similarities, index = tags.index)
        return similarities

    def recommend(self):
        similarities = self.calcSimilarity()
        similarities = similarities.drop(index = self.disliked)
        recommended = {user:self.users[user].recommend(similarities, i+1) for i,user in enumerate(self.users)}
        return recommended
    
    def addLiked(self, movieId, userId):
        self.tags.loc[movieId, 'liked'] += 1/(self.n_users-1)
        self.liked += [movieId]
        self.users[userId].liked += [movieId]
        self.all += [movieId]
        self.users[userId].all += [movieId]
        
    def addDisliked(self, movieId, userId):
        self.tags.loc[movieId, 'liked'] -= 1/(self.n_users-1)
        self.disliked += [movieId]
        self.users[userId].disliked += [movieId]
        self.all += [movieId]
        self.users[userId].all += [movieId]
        
    def addSeen(self, movieId, userId):
        self.tags.loc[movieId, 'liked'] += 1/(self.n_users-1)
        self.seen += [movieId]
        self.users[userId].seen += [movieId]
        self.all += [movieId]
        self.users[userId].all += [movieId]
        
    def match(self):
        if self.n_users > 1:
            common = set(self.users[self.ids[0]].liked) & set(self.users[self.ids[1]].liked)
            for i in range(2, self.n_users):
                common = common & set(self.users[self.ids[i]].liked)
            return common.pop() if common else False
        
        return self.users[ids[0]].liked[0] if self.users[ids[0]].liked else False

    
class User:
    def __init__(self):
        self.liked = []
        self.disliked = []
        self.seen = []
        self.all = []
    
    def recommend(self, similarities, n = 1):
        similarities = similarities.drop(index = self.liked + self.seen, errors = 'ignore')
        return similarities.nlargest(n, 0).index.values[-1]


