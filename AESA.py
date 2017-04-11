import numpy as np
import math


class AESA():
    def __init__(self, train_data):
        self.train_data = train_data
        self.distances = self.createDistMatrix(train_data)
        self.data_size = np.shape(train_data)

    def createDistMatrix(self, data):
        return [[self.eucDist(x1, x2) for x1 in data] for x2 in data]

    def eucDist(self, x1, x2):
        return np.sqrt(sum((x1-x2)**2))

    def getNN(self, q):

        lower = np.zeros(self.data_size[0])

        candidate_pool = list(range(self.data_size[0]))

        while candidate_pool:

            selected = min(candidate_pool, key=lambda i: lower[i])

            current_dist = self.eucDist(q, self.train_data[selected])

            best_dist = math.inf

            if current_dist < best_dist:
                best = selected
                best_dist = current_dist

            last_candidate_pool = candidate_pool
            candidate_pool = []

            for i in last_candidate_pool:

                bounds = abs(current_dist - self.distances[selected][i])
                lower[i] = max(lower[i], bounds)

                if lower[i] < best_dist:
                    candidate_pool.append(i)

        return self.train_data[best]


gaussian_data = np.random.normal(2.4, 0.5, 200).reshape(100, 2)

if __name__ == '__main__':
    import sys

    if 'test' in sys.argv:

        gaussian_data = np.random.normal(2.4, 0.5, 200).reshape(100, 2)

        aesa = AESA(gaussian_data)
        query = [2,  2]
        print(aesa.getNN(query))
