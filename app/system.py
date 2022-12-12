import pandas as pd
import numpy as np
import json


df_ratings = pd.read_csv('ratings.dat',header=None, index_col=False, sep="::", names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# movie and their info(name)
with open('movies.json') as json_file:
    movies_dict = json.load(json_file)
with open('columns.json') as json_file:
    columns = json.load(json_file)
with open('most_popular_movie.json') as json_file:
    most_popular_movie_dict = json.load(json_file)

rated_movies = dict()
for c in columns['columns']:
    c = str(c)
    rated_movies[c] = movies_dict[c]

genre_set = {'Action',
 'Adventure',
 'Animation',
 "Children's",
 'Comedy',
 'Crime',
 'Documentary',
 'Drama',
 'Fantasy',
 'Film-Noir',
 'Horror',
 'Musical',
 'Mystery',
 'Romance',
 'Sci-Fi',
 'Thriller',
 'War',
 'Western'}

def listOfTuples(l1, l2):
    return list(map(lambda x, y:(x,y), l1, l2))

def system1(genre):
    g = dict()
    for i in most_popular_movie_dict[genre]:
        i = str(i)
        g[i]=movies_dict[i]
    return g



def system2(ls):
    test_ubcf = pd.DataFrame([ls],columns=columns['columns'])
    
    data_ubcf=df_ratings
    data_ubcf = data_ubcf.pivot_table("Rating","UserID","MovieID",aggfunc="first")
    p_train_ubcf = data_ubcf
    p_train_ubcf = p_train_ubcf.sub(p_train_ubcf.mean(axis=1), axis=0)


    test_nan_ls = test_ubcf.columns[test_ubcf.isna().any()].tolist()
    test_ubcf = test_ubcf.loc[:, ~test_ubcf.columns.isin(test_nan_ls)]
    m = test_ubcf.mean(axis=1)
    test_ubcf = test_ubcf.sub(m, axis=0)
    train_ubcf = p_train_ubcf.loc[:, ~p_train_ubcf.columns.isin(test_nan_ls)]
    df_column_test = pd.melt(test_ubcf[0:1],value_vars=test_ubcf.columns)
    watched = set(df_column_test['variable'])

    cos_sim = []
    no_similarity = []
    for i in range(6040):
        xy = 0
        x = 0
        y = 0
        df_row = train_ubcf[i:i+1]

        df_column = pd.melt(df_row,value_vars=train_ubcf.columns)
        if df_row.isna().sum().sum()==len(test_ubcf.columns):
            cos_sim.append(np.nan)
            no_similarity.append(i+1)
            continue
        for j in range(len(test_ubcf.columns)):
            if pd.isnull(df_column['value'][j]):
                pass
            else:
                y += (df_column_test['value'][j])**2
                x += (df_column['value'][j])**2
                xy += df_column['value'][j]*df_column_test['value'][j]
        cos_sim.append(xy/((x**0.5)*(y**0.5)))
    df_cos_sim = pd.DataFrame({'UserID':train_ubcf.index,'similarity':cos_sim})
    df_cos_sim['similarity']=(1+df_cos_sim['similarity'])/2
    df_cos_sim_users = df_cos_sim.sort_values('similarity',ascending=False)[:20]
    df_cos_sim_users_ratings = pd.concat([df_cos_sim_users, p_train_ubcf], axis=1, join="inner")
    df_cos_sim_users_ratings2 = df_cos_sim_users_ratings
    df_cos_sim_users_ratings2 = df_cos_sim_users_ratings2.notnull().astype("int")
    mypred = []
    for i in df_cos_sim_users_ratings2.columns:
        if i == 'UserID' or i=='similarity':
            continue
        temp = df_cos_sim_users['similarity']*df_cos_sim_users_ratings[i]
        temp2 = df_cos_sim_users['similarity']*df_cos_sim_users_ratings2[i]
        n1 = temp.sum()
        n2 = temp2.sum()
        if i in watched:
            mypred.append(np.nan)
        else:
            mypred.append((n1/n2)+m[0])
    result = listOfTuples(list(df_cos_sim_users_ratings2.columns)[2:], mypred)
    result = [i for i in result if pd.isna(i[1])!=True]
    result = sorted(result, key=lambda x: x[1])
    
    if len(result)>=5:
        movie_ls =  [i for i,j in result[-5:]]
    else:
        movie_ls = [1,2,3,4,5]

    g = dict()
    for k in movie_ls:
        k = str(k)
        g[k]=movies_dict[k]
    return g



    






