import scipy.sparse as sp
import numpy as np

class Dataset(object):
    def __init__(self, path):
        self.trainMatrix = self.loadRatingFileAsMatrix(path + '.train.rating')
        self.testRatings = self.loadRatingFileAsList(path + '.test.rating')
        self.testNegatives = self.loadNegativeFile(path + '.test.negative')
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def loadRatingFileAsMatrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename,'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u,i = int(arr[0]), int(arr[1])
                num_users = max(num_users,u)
                num_items = max(num_items,i)
                line= f.readline()

        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename,'r') as f:
            
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])

                if rating > 0.0:
                    mat[user,item] = 1.0
                line = f.readline()
        return mat

    def loadRatingFileAsList(self, filename):
        ratingList = []
        with open(filename,'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                user,item = int(arr[0]), int(arr[1])
                ratingList.append([user,item])
                line = f.readline()
        return ratingList

    def loadNegativeFile(self, filename):
        negativeList = []
        with open(filename,'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
