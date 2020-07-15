from knn_from_scratch import knn,euclidean_distance

def recommend_movies(query,num_of_recommendation):
    raw_movies_data = []
    with open('movies_recommendation_data.csv','r') as md:
        # Discard the first line which is the heading
        next(md)

        for line in md.readlines():
            data_row = line.strip().split(',')
            raw_movies_data.append(data_row)
        
        movies_recommmendation_data = []
        for row in raw_movies_data:
            data_row = list(map(float,row[2:]))
            movies_recommmendation_data.append(data_row)
        # Use knn to get the k movie recommendation
        recommend_indices,_ =  knn(movies_recommmendation_data,query,num_of_recommendation,euclidean_distance,lambda x: None)

        movie_recommendations = []
        for _,index in recommend_indices:
            movie_recommendations.append(raw_movies_data[index])

        return movie_recommendations
        
if __name__ == '__main__':
    # feature vector (the movie we selected to see movies like this)
    the_post = [7.2, 1, 1, 0, 0, 0, 0, 1, 0]
    recommended_movies = recommend_movies(query=the_post,num_of_recommendation=5) # recommend 5 movie similar to the post

    for movie in recommended_movies:
        print(movie[1])