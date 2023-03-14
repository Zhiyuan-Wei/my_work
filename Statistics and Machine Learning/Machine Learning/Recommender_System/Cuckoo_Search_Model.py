import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import random


## parameters
group_no = 4
Pa = 0.5
ite = 1
raw_train_1_Pa = 0.6
raw_train_2_Pa = 1-0.2/(1-raw_train_1_Pa)
file_path = 'C:\\Users\\Zhiyuan\\Desktop\\沃顿商学院\\project\\ratings_small.csv'



rmse_array = np.array([])
for group_no in tqdm([4,8,12,16,24,32,48,64]):
    raw = pd.read_csv(file_path)
    raw.set_index(['userId','movieId'],inplace = True)
    raw = raw.unstack().rating.fillna(0)
    raw_origin = raw.copy(deep = True)
    film_index = raw_origin.columns
    raw['location'] = raw.apply(lambda x: np.array(x),axis = 1)

    # divide users into Ua(raw_train_1), Ub(raw_train_2) and test group(raw_test)
    raw_train_1_df = raw_origin.sample(frac = raw_train_1_Pa)
    raw_train_1 = raw_train_1_df.copy(deep = True)
    raw_train_1 = np.array(raw_train_1)

    raw_origin = raw_origin.append(raw_train_1_df)
    difference_set_result_1 = raw_origin.drop_duplicates(film_index,keep=False)

    raw_train_2 = difference_set_result_1.sample(frac = raw_train_2_Pa)
    raw_train_2_df = raw_train_2.copy(deep = True)
    raw_train_2 = np.array(raw_train_2)

    difference_set_result_1 = difference_set_result_1.append(raw_train_2_df)
    raw_test = difference_set_result_1.drop_duplicates(film_index,keep=False)
    raw_test_df = raw_test.copy(deep = True)


    # k-means
    clf = KMeans(n_clusters=group_no)
    clf.fit(raw_train_1)

    centers = clf.cluster_centers_
    labels = clf.labels_


    # get location, cluster number, center location and distance to center of the users in Ua
    raw_train_1_df['location'] = raw_train_1_df.apply(lambda x: np.array(x),axis = 1)
    raw_train_1_df['cluster'] = labels
    raw_train_1_df['centers'] = raw_train_1_df['cluster'].apply(lambda x: centers[x])
    raw_train_1_df['distance'] = raw_train_1_df.apply(lambda x: np.sqrt(np.sum(np.power(x['location']-x['centers'],2))),axis = 1)


    ## calculate ave_distance of each cluster
    ave_distance = np.array([])
    for i in range(group_no):
        ave_distance = np.append(ave_distance, np.mean(raw_train_1_df[raw_train_1_df['cluster'] == i]['distance']))

    total_df = pd.concat([raw, raw_train_1_df[['cluster','centers','distance']]],axis = 1,sort = True)

    
    ## add Ub users to training group
    for i in range(len(raw_train_2_df)):
        distance_list = np.array([])
        distance_ratio_list = np.array([])
        for j in range(len(centers)):
            distance = np.sqrt(np.sum(np.power(np.array(raw_train_2_df.loc[raw_train_2_df.index[i]]) - centers[j],2)))
#            distance_ratio = distance/ave_distance[j]
            distance_list = np.append(distance_list,distance)
            distance_ratio_list = np.append(distance_ratio_list,distance/ave_distance[j])

        total_df.loc[raw_train_2_df.index[i],'cluster'] = np.argmin(distance_ratio_list)
        total_df.loc[raw_train_2_df.index[i],'distance'] = distance_list[np.argmin(distance_ratio_list)]

        if total_df.loc[raw_train_2_df.index[i],'distance'] < np.mean(total_df[total_df['cluster'] == np.argmin(distance_ratio_list)]['distance']):

            index = np.argmax(total_df[total_df['cluster'] == np.argmin(distance_ratio_list)]['distance'])
            userId = total_df[total_df['cluster'] == np.argmin(distance_ratio_list)].index[index]
            rand = random.random()
            if rand < Pa:
                total_df.drop(total_df[total_df.index == userId].index, inplace=True)
            else:
                continue
        else:
            total_df.drop(total_df[total_df.index == raw_train_2_df.index[i]].index, inplace = True)


    # calculate average marks in each cluster and get mark_list_nan
    mark_list_nan = pd.DataFrame()
    mark = total_df.iloc[:,:-4]
    mark = mark.replace(0,np.nan)
    mark = pd.concat([mark,total_df['cluster']],axis = 1)
    for i in range(group_no):
        mark_list_nan[i] = mark[mark['cluster'] == i].iloc[:,:-1].mean()


    # reset the center of each cluster and get centers_new
    mark_list = pd.DataFrame()
    mark = total_df.iloc[:,:-4]
    mark = pd.concat([mark,total_df['cluster']],axis = 1)
    for i in range(group_no):
        mark_list[i] = mark[mark['cluster'] == i].iloc[:,:-1].mean()
    centers_new = np.array(mark_list.T)
    centers_new_2 = np.array(mark_list_nan.T)


    # renew total_df
    total_df_new = total_df.iloc[:,:-4]
    total_df_new = total_df_new.append(raw_test_df)
    total_df_new = total_df_new.drop_duplicates(film_index,keep=False)

    total_df_new['location'] = total_df_new.apply(lambda x: np.array(x),axis = 1)
    total_df_new['cluster'] = total_df['cluster']
    total_df_new['centers'] = total_df_new['cluster'].apply(lambda x: centers_new[int(x)])
    total_df_new['distance'] = total_df_new.apply(lambda x: np.sqrt(np.sum(np.power(x['location']-x['centers'],2))),axis = 1)

    
    ## calculate ave_distance of each cluster
    ave_distance = np.array([])
    for i in range(group_no):
        ave_distance = np.append(ave_distance, np.mean(raw_train_1_df[raw_train_1_df['cluster'] == i]['distance']))

    total_df_new = pd.concat([raw, total_df_new[['cluster','centers','distance']]],axis = 1,sort = True)

    
    # calculate distance of each test user to according cluster center
    for i in range(len(raw_test_df)):
        distance_list = np.array([])
        distance_ratio_list = np.array([])
        for j in range(len(centers_new)):
            distance = np.sqrt(np.sum(np.power(np.array(raw_test_df.loc[raw_test_df.index[i]]) - centers_new[j],2)))
#            distance_ratio = distance/ave_distance[j]
            distance_list = np.append(distance_list,distance)
            distance_ratio_list = np.append(distance_ratio_list,distance/ave_distance[j])

        total_df_new.loc[raw_test_df.index[i],'cluster'] = np.argmin(distance_ratio_list)
        total_df_new.loc[raw_test_df.index[i],'distance'] = distance_list[np.argmin(distance_ratio_list)]


    # calculate RMSE
    raw_test_result = total_df_new.loc[raw_test_df.index]
    raw_test_result['RMSE'] = raw_test_result.apply(lambda x: np.array(x['location']),axis = 1)
    for i in raw_test_result.index:
        location_nan = np.array([np.nan if j==0 else j for j in raw_test_result.loc[i,'location']])
        center_nan = np.array([np.nan if j==0 else j for j in centers_new_2[int(raw_test_result.loc[i,'cluster'])]])
        raw_test_result.loc[i,'RMSE'] = np.sqrt(np.mean(np.power((location_nan -center_nan)[(location_nan -center_nan <11)],2)))
    rmse_array = np.append(rmse_array, np.mean(np.abs(raw_test_result['RMSE'])))


print(np.mean(rmse_array))
