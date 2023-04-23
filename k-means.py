import numpy as np
import cv2

class K_Means:
    '''
    That's a self implemented class for K-Means clustering algorithm
    '''
    num_center = 0
    data_dimension = 0
    data = None
    z = None
    c = None
    termination = False
    G = None

    def __init__(self, k, dimension, data) -> None:
        self.num_center = k
        self.data_dimension = dimension ** 2
        self.data = data
        self.z = k*[None]
        self.c = len(data)*[None]

    def dist(self, point1, point2) -> int:
        '''
        calculates the Euclidean distance between two points
        '''
        sum = 0
        for i in range(self.data_dimension):
            sum += (int(point2[0][i]) - int(point1[0][i]))**2
        return np.sqrt(sum)
    
    def random_init_center(self):
        '''
        This function initialize the centers for start of the algorithm
        '''
        temp_list = np.random.choice(len(self.data), size= self.num_center, replace= False)
        for i, j in zip(range(self.num_center), temp_list):
            self.z[i] = self.data[j]

    def clustering(self):
        '''
        This function will clusters data based on Euclidean distance
        '''
        min_dist = self.dist(self.data[0], self.z[0])
        min_ind = 0
        self.G = [{} for i in range(self.num_center)]
        for i_ind, i in enumerate(self.data):
            for j_ind, j in enumerate(self.z):
                dist = self.dist(i, j)
                if min_dist > dist:
                    min_dist = dist
                    min_ind = j_ind
            self.c[i_ind] = min_ind
            self.G[min_ind][i_ind] = min_dist
            if i_ind != len(self.data) - 1: min_dist = self.dist(self.data[i_ind + 1], self.z[0])
            min_ind = 0
    
    def Update_Centers(self):
        '''
        This function updates the centers of each cluster
        '''
        self.termination = True
        for j in range(self.num_center):
            temp = self.z[j]
            self.z[j] = np.zeros((1, self.data_dimension))
            num_of_data = 0
            for i in range(len(self.data)):
                if self.c[i] == j:
                    self.z[j] += self.data[i]
                    num_of_data += 1
            self.z[j] /= num_of_data
            if (self.z[j] != temp).all(): self.termination = False

    def Show_Samples(self):
        '''
        This function shows 20 samples of each of the Clusters
        '''
        temp = np.zeros((1, self.num_center))
        for j in range(self.num_center):
            print(j + 1, "Cluster: ")
            cv2.imwrite(f'Cluster {j + 1}/Pivot{j}.jpg', self.z[j].reshape((16, 16)))
            for i in range(len(self.data)):
                if self.c[i] == j:
                    if temp[0][j] < 20:
                        temp[0][j] += 1
                        cv2.imwrite(f'Cluster {j + 1}/image{temp[0][j]}.jpg', self.data[i].reshape((16, 16)))
                        # cv2.imshow('image', self.data[i].reshape((16, 16)))
                        # cv2.waitKey(0)
                        if temp[0][j] == 20: break

    def Show_Samples_KNN(self, n):
        '''
        This function shows n nearest samples to each center of the Clusters
        '''
        for j in range(self.num_center):
            self.G[j] = dict(sorted(self.G[j].items(), key=lambda item:item[1]))
            print(j + 1, "Cluster: ")
            cv2.imwrite(f'Cluster {j + 1}/Pivot{j}.jpg', self.z[j].reshape((16, 16)))
            num = 0
            for i in self.G[j]:
                if num < n:
                    cv2.imwrite(f'Cluster {j + 1}/image{num}.jpg', self.data[i].reshape((16, 16)))
                    num += 1
                else: break

    def Do(self):
        '''
        This function automatically does the K-Means algorithm and prints 20 samples
        '''
        num = 0
        self.random_init_center()
        while not self.termination:
            self.clustering()
            self.Update_Centers()
            print("-----------------", num, "-----------------")
            num += 1
        self.Show_Samples()


    def Do_KNN(self, KNN):
        '''
        This function automatically does the K-Means algorithm and prints n nearest neighbors
        '''
        num = 0
        self.random_init_center()
        while not self.termination:
            self.clustering()
            self.Update_Centers()
            print("-----------------", num, "-----------------")
            num += 1
        self.Show_Samples_KNN(KNN)


A = cv2.imread("F:/University/6th Term/Linear Algebra/2/f14b2_3077/USPS_1_5/usps_1.jpg", 0)
B = cv2.imread("F:/University/6th Term/Linear Algebra/2/f14b2_3077/USPS_1_5/usps_2.jpg", 0)
C = cv2.imread("F:/University/6th Term/Linear Algebra/2/f14b2_3077/USPS_1_5/usps_3.jpg", 0)
D = cv2.imread("F:/University/6th Term/Linear Algebra/2/f14b2_3077/USPS_1_5/usps_4.jpg", 0)
E = cv2.imread("F:/University/6th Term/Linear Algebra/2/f14b2_3077/USPS_1_5/usps_5.jpg", 0)

data = []
dimension = 16

for i in range(34):
    for j in range(33):
        digital_img = A[i * dimension: (i + 1) * dimension, j * dimension: (j + 1) * dimension]
        if (not(i > 11 and j == 32)):
            data.append(digital_img.reshape(1, dimension ** 2))

        digital_img = B[i * dimension: (i + 1) * dimension, j * dimension: (j + 1) * dimension]
        if (not(i > 11 and j == 32)):
            data.append(digital_img.reshape(1, dimension ** 2))

        digital_img = C[i * dimension: (i + 1) * dimension, j * dimension: (j + 1) * dimension]
        if (not(i > 11 and j == 32)):
            data.append(digital_img.reshape(1, dimension ** 2))
        
        digital_img = D[i * dimension: (i + 1) * dimension, j * dimension: (j + 1) * dimension]
        if (not(i > 11 and j == 32)):
            data.append(digital_img.reshape(1, dimension ** 2))
        
        digital_img = E[i * dimension: (i + 1) * dimension, j * dimension: (j + 1) * dimension]
        if (not(i > 11 and j == 32)):
            data.append(digital_img.reshape(1, dimension ** 2))

k = 5
#num_samples = 20

Run = K_Means(k, dimension, data)
# edit
print("Hello World")

Run.Do()
