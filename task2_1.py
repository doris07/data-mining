from pyspark import SparkContext, SparkConf
import math
import sys
import time
from itertools import combinations
from operator import add
import operator
import csv

#Case 1: Item-based CF recommendation system with Pearson similarity
#https://blog.csdn.net/moonfanslth/article/details/82078989
#Pearson similarity is a measure of the correlation between two sets of data
#which can be used to calculate the similarity between items in a recommendation system
#collect data&build item-item matrix
#calculate pearson similarity
#make recommendations
#evaluate the system

train_path = sys.argv[1] #yelp_train.csv
val_path = sys.argv[2] #yelp_val.csv
outputFile_path = sys.argv[3] #prediction results for each user and business pair 
#===============
#https://blog.csdn.net/sinat_22594309/article/details/86420207


def item_based_prediction(testdt, user_id, business_id, user_business_dict, business_mean_dict, business_rating_dicts, neighbourhood_max):
    users_val = testdt[0] #user_id
    businesses_val = testdt[1] #business_id
    #cold start def:
    if users_val not in user_id.keys():
        return (users_val, businesses_val, 2.5)  #new user
    if businesses_val not in business_id.keys():
        return (users_val, businesses_val, 2.5) #new business
    if users_val in user_id.keys():
        if businesses_val in business_id.keys():
            if users_val in business_id[businesses_val]:
                return  (users_val, businesses_val, user_business_dict[(users_val, businesses_val)])
    #co-rated item
    users_res = user_id[users_val]
    #users_res = find_item(users_val, businesses_val,user_id, business_id)
    businesses_res = business_id[businesses_val]
    businesses_mean_res = business_mean_dict[businesses_val]
    #check debug
    #print('\nuser_business dict length:\n',len(list(users_res)))
    #print('users with: \n{}'.format(users_res))
    #print('business with rating: \n{}'.format(businesses_res))
    #print('\nuser_business dict length:\n',len(list(businesses_res)))
    #print(users_val)

    ratinglists = []
    for curr_caiji in users_res:
        corated_itemlist = list(set(business_id[curr_caiji]).intersection(set(businesses_res)))
        comb = user_business_dict[users_val, curr_caiji]
        if len(corated_itemlist) ==1 or len(corated_itemlist) == 0:
            Diff = business_mean_dict[curr_caiji] - businesses_mean_res
            if (Diff <= 1):
                pearson_coe = 1.0
                ratinglists.append([curr_caiji, comb, pearson_coe]) 
            else:
                pearson_coe = 0.0
                ratinglists.append([curr_caiji,comb, pearson_coe])
        else:
            numerator = 0
            denominator_caiji = 0
            denominator_val = 0
            a0 = business_rating_dicts[curr_caiji]
            #print('aaaa\n',a0)
            a1 = business_rating_dicts[businesses_val]
            #print('\nbbbbbb\n',a1)
            val_rating_dict = list(map( a0.get, corated_itemlist))
            #print('\naaaa-11111:',val_rating_dict[:2],'\n')
            businesses_rating_dict = list(map(a1.get, corated_itemlist))
            #print('\nbbbbb-11111:',val_rating_dict[:2],'\n')
            #mean = business_mean_dict[curr_caiji]
            for i in range(len(corated_itemlist)):
                numerator += (val_rating_dict[i] - business_mean_dict[curr_caiji]) * (businesses_rating_dict[i] - business_mean_dict[businesses_val])
                denominator_caiji += (val_rating_dict[i] - business_mean_dict[curr_caiji]) ** 2
                denominator_val += (businesses_rating_dict[i] - business_mean_dict[businesses_val]) ** 2
            denominator_res = math.sqrt(denominator_caiji) * math.sqrt(denominator_val)
            if (denominator_res == 0):
                pearson_coe = 0.0
            else: 
                pearson_coe = float(numerator/denominator_res)
            ratinglists.append([curr_caiji, comb, pearson_coe]) if pearson_coe >= 0 else None
            if len(businesses_res) != 0:
                pearson_simi_0 = []
        #ratinglists = find_pearson_itemlist(users_res, users_val, businesses_val, business_id, businesses_res, business_mean_dict, businesses_mean_res, user_business_dict, business_rating_dicts)
        #print('\n\n similarity-final:\n',pearson_simi)
    pearson_simi = sorted(ratinglists, key=lambda x: x[2], reverse=True)
    neighbourhood_max = min(neighbourhood_max,len(pearson_simi)) #50
    weight_sum = 0
    for xcj in range(len(pearson_simi[:neighbourhood_max])):
        weight_sum += pearson_simi[xcj][1] * pearson_simi[xcj][2]
    w = list(map(lambda tup: tup[2], pearson_simi[:neighbourhood_max]))
    if sum(w) == 0:
        return (users_val, businesses_val, 2.5)
    normalized_w = sum(w)
    preditions = (weight_sum / normalized_w)
    res = (users_val, businesses_val, preditions)
    #print('\n\n final:\n',res)
    return (users_val, businesses_val, preditions)

#=================
conf = SparkConf()
conf.setMaster("local[*]").setAppName("hw3-task2-case1") #set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
#Use SparkContext library and read file as text and then map it to json.
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

start_time = time.time()
#############collect data
train_dtRDD = sc.textFile(train_path)
train_header = train_dtRDD.first()
dtRDD_train = train_dtRDD.filter(lambda a: a != train_header)
train_dt = dtRDD_train.map(lambda x: (x.split(',')))

#find out all the unique users in the yelp_train.csv:
#prepare the data for co-rated items between user
user = train_dt.map(lambda x: (x[0],x[1])).groupByKey().sortByKey(True).mapValues(list).sortByKey()
business = train_dt.map(lambda x: (x[1],x[0])).groupByKey().sortByKey(True).mapValues(list).sortByKey()
user_business =train_dt.map(lambda x: ((x[0], x[1]), float(x[2])))
#==========================
user_dict = dict(user.collect()) #user_id dictonary
business_dict = dict(business.collect()) #business id dictonary
user_business_dict = dict(user_business.collect()) #(user_id,business) dictonary
#==========================
#[('a',{'','','',''}),('b',{'','','',''})]
business_rating = train_dt.map(lambda x: (x[1],(x[0], float(x[2])))).groupByKey().mapValues(dict)
business_rating_dicts = dict(business_rating.collect())
user_rating = train_dt.map(lambda x: (x[0],(x[1],float(x[2])))).groupByKey().mapValues(dict)
user_rating_dict = dict(user_rating.collect())
#=============== average prepare:
business_mean = train_dt.map(lambda x: (x[1], (float(x[2]), 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda x: x[0]/x[1])
business_mean_dict = dict(business_mean.collect())

test_dtRDD = sc.textFile(val_path)
test_header = test_dtRDD.first()
dtRDD_test = test_dtRDD.filter(lambda a: a != test_header)
test_dt = dtRDD_test.map(lambda x: (x.split(',')))
N = 50 
prediction_res = test_dt.map(lambda x: item_based_prediction(x, user_dict, business_dict, user_business_dict, business_mean_dict, business_rating_dicts, N)).collect()
#print(prediction_res[:5])

with open(outputFile_path,'w', newline= '') as outfile:
    dt_as_csv = csv.writer(outfile)
    dt_as_csv.writerow(['business_id_1', 'business_id_2', 'similarity'])
    for item in prediction_res:
        dt_as_csv.writerow([item[0],item[1],item[2]])

end_time = time.time()
exe_time = end_time - start_time
print("“Duration: {}".format(exe_time))

#output_dict = {}
#with open(outputFile_path, 'r') as f:
#    reader = csv.reader(f)
#    next(reader)
#    for row in reader:
#        output_dict[(row[0], row[1])] = float(row[2])

#test_dict = test_dt.map(lambda x: ((x[0], x[1]), float(x[2]))).collectAsMap()
#se = 0
#for k, v in test_dict.items():
#    if k in output_dict:
#        se += (v - output_dict[k]) ** 2
#rmse = math.sqrt(se / len(test_dict))
#print("RMSE:", rmse)

#23/03/21 11:14:18 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
#23/03/21 11:14:19 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
#“Duration: 64.75798392295837                                                    
#RMSE: 1.0698210788878666