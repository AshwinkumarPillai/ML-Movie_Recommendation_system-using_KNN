from collections import Counter
import math

# data is the training data and query is the list of height for which we have to predict the weight.
# distance_fn is the function we will use to calculate the distance between neighbors
# choice_fn is mean for regression and mode for classification
def knn(data,query,k,distance_fn,choice_fn):
    if k == 0:
        k=3
    neighbor_distance_and_indices = []
    # for each example in the data
    for index,example in enumerate(data):
        # calc the distance between the query and the current example
        distance = distance_fn(example[:-1],query)
        neighbor_distance_and_indices.append((distance,index))

    sorted_dist_and_indices = sorted(neighbor_distance_and_indices)

    # get the first k nearest neighbors and their respective labels
    k_nearest_distance_and_indices = sorted_dist_and_indices[:k]
    k_nearest_labels = [data[i][1] for distance,i in k_nearest_distance_and_indices]
    
    return k_nearest_distance_and_indices, choice_fn(k_nearest_labels)

def mean(data):
    return sum(data) / len(data)

def mode(data):
    return Counter(data).most_common(1)[0][0]

def euclidean_distance(points1,points2):
    sum_squared_distance = 0
    for i in range(len(points1)):
        sum_squared_distance += math.pow(points1[i] - points2[i],2)
    return math.sqrt(sum_squared_distance)

def main():
    # data = [height,weight]
    
    # data  for regression
    reg_data = [
       [65.75, 112.99],
       [71.52, 136.49],
       [69.40, 153.03],
       [68.22, 142.34],
       [67.79, 144.30],
       [68.70, 123.30],
       [69.80, 141.49],
       [70.01, 136.46],
       [67.90, 112.37],
       [66.49, 127.45],
    ]

    # for given height what is the weight; height = 60
    reg_query = [60]
    reg_k_nearest_neighbors, reg_prediction = knn(reg_data, reg_query,k=3,distance_fn=euclidean_distance,choice_fn=mean)
    print(reg_prediction)
    # data = [age,likes brinjal]

    # data for classification
    clf_data = [ 
       [22, 1],
       [23, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
    ]

    # given the age tell whether the person likes brinjal or not
    clf_query = [20]
    clf_k_nearest_neighbors, clf_prediction = knn(clf_data, clf_query, k=3, distance_fn=euclidean_distance,choice_fn=mode)
    print(clf_prediction)
if __name__ == '__main__':
    main()