import os

import pandas as pd 
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html



data_movie = pd.read_csv("./dataSet/movie.csv")
data_rating = pd.read_csv("./dataSet/rating.csv")

# 1. preprocessing
# user가 너무 많으므로 1000명으로 줄임 
data_rating_small = data_rating[data_rating['userId']<=1000]

# 2. making rating matrix 
merges = pd.merge(data_rating_small,data_movie,on="movieId",how="left")
merges.drop("timestamp",axis=1,inplace=True)

rating_matrix= merges.pivot_table('rating', index = 'title', columns='userId')
rating_matrix = rating_matrix.fillna(0)

# 3. function
'''
function : collaborative Filtering by item
input
    -item(str) : item name to be compared (movie name)
    -N(int) : default 10
              ordering nubmer
'''
def itembasedResult(item,N=10):
    #similarity
    rm = rating_matrix.copy(deep=True)
    sim_item = cosine_similarity(rm)

    #order
    title = rm.index.to_numpy()

    title_index = np.where(title==item)[0][0]

    ar = sim_item[title_index]

    order_index = np.argsort(-ar)

    result = []
    cnt = 1
    for i in order_index:
        if(cnt<=N):
            print(cnt,end="\t")
            print(i,end="\t")
            print(sim_item[title_index][i],end="\t")
            print(title[i])
            result.append([i,title[i]])
            cnt+=1
    #return result


'''
function : collaborative Filtering by user
input
    -personId(INT) : user to be compared (userId)
'''
#- user id별 영화 추천 ordering list
def userbasedResult(personId):
    rm = rating_matrix.copy(deep=True)
    #cos sim  by user
    sim_user = cosine_similarity(rm.T)
    #- 처리 편하게 하기위해서 (x축 -> id , y축 -> 영화 ) === ppt 그림 형식
    rating_np = rm.to_numpy()
    rating_np = rating_np.T

    for i in range(len(sim_user)):
        for j in range(len(rating_np[0])):
            rating_np[i][j] = rating_np[i][j]*sim_user[personId-1][i]
    #다시 제자리로
    rating_np = rating_np.T
    person_rating = pd.DataFrame(rating_np)
    person_rating.index = rating_matrix.index
    person_rating.columns = rating_matrix.columns

    #결과
    movie_rating = pd.DataFrame(person_rating.sum(axis=1))
    movie_rating.columns = [personId]

    #결과(ordering) 
    movie_rating = movie_rating.sort_values(by=[20],ascending=False)
    return movie_rating


# 4. result
itembasedResult("Zombie (a.k.a. Zombie 2: The Dead Are Among Us) (Zombi 2) (1979)")
userbasedResult(20)


