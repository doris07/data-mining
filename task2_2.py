from pyspark import SparkContext, SparkConf
import sys
import csv
import json
import time
import math
import numpy as np
import xgboost as xgb


##Model-based recommendation system
train_path = sys.argv[1] #folder path
val_path = sys.argv[2] #yelp_val.csv
outputFile_path = sys.argv[3] #

conf = SparkConf()
conf.setMaster("local[*]").setAppName("hw3-task2-2") #set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
#Use SparkContext library and read file as text and then map it to json.
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

def before_merge(data):
    dtRDD = sc.textFile(data)
    header = dtRDD.first()
    dtRDD_final = dtRDD.filter(lambda a: a != header)
    final_data = dtRDD_final.map(lambda x: (x.split(',')))
    return final_data

def more_data(data1, data2):
    dtRDD_1 = sc.textFile(data1)
    datasets = dtRDD_1.map(lambda a: json.loads(a))
    #{"business_id":"Apn5Q_b6Nz61Tq4XzPdf9A", "stars":4.0, "review_count":24,{"RestaurantsPriceRange2":"2"}}
    #{"business_id":"AjEbIBw6ZFfln7ePHha9PA","stars":4.5,"review_count":3,"RestaurantsPriceRange2":"2"}
    businesses = datasets.map(lambda a: ((a['business_id'], 
                                          (a['stars'], a['review_count'], a['attributes'].get('RestaurantsPriceRange2') if a['attributes'] else None))))
    #businesses = datasets.map(lambda a: ((a['business_id'], 
    #                                      (a['stars'], a['review_count']))))

    dtRDD_2 = sc.textFile(data2)
    dataset = dtRDD_2.map(lambda a: json.loads(a))
    #{"user_id":"s4FoIXE_LSGviTHBe8dmcg","review_count":3,"average_stars":3.0}
    #{"user_id":"ZcsZdHLiJGVvDHVjeTYYnQ","review_count":3,"average_stars":5.0,}
    #{"user_id":"h3p6aeVL7vrafSOM50SsCg","review_count":2,"average_stars":5.0}
    user_dt = dataset.map(lambda a: ((a['user_id'], (a['review_count'], a['average_stars']))))
    return businesses, user_dt

def merge_train_data(dataset, businesses, users):
    users_id = dataset[0] #user_id
    businesses_id = dataset[1] #business_id
    stars = float(dataset[2])
    #merge data
    if businesses_id not in businesses.keys(): #failed to find matched business_id
        if users_id not in users.keys(): #failed to find matched user_id
            return [users_id, businesses_id, None, None, None, None, None, stars]
        elif users_id in users.keys():
            review_count, average_stars = users[users_id]
            return [users_id, businesses_id, review_count,average_stars, None, None, None, stars]
        
    if businesses_id in businesses.keys():
        review_stars, review_counts, RestaurantsPriceRange2  = businesses[businesses_id]
        if users_id not in users.keys():
            return [users_id, businesses_id, None, None, review_stars, review_counts, float(RestaurantsPriceRange2) if RestaurantsPriceRange2 is not None else None, stars]
        elif users_id in users.keys():
            review_count, average_stars = users[users_id]
            return [users_id, businesses_id, review_count,average_stars, review_stars,review_counts,float(RestaurantsPriceRange2) if RestaurantsPriceRange2 is not None else None, stars]

def merge_test_data(dataset, businesses, users):
    users_id = dataset[0] #user_id
    businesses_id = dataset[1] #business_id
    #stars = float(dataset[2])
    #merge data
    if businesses_id not in businesses.keys(): #failed to find matched business_id
        if users_id not in users.keys(): #failed to find matched user_id
            return [(users_id, businesses_id), None, None, None, None, None]
        elif users_id in users.keys():
            review_count, average_stars = users[users_id]
            return [(users_id, businesses_id), review_count, average_stars, None, None, None]
    if businesses_id in businesses.keys():
        review_stars, review_counts, RestaurantsPriceRange2  = businesses[businesses_id]
        if users_id not in users.keys():
            return [(users_id, businesses_id), None, None, review_stars, review_counts, float(RestaurantsPriceRange2) if RestaurantsPriceRange2 is not None else None]
        elif users_id in users.keys():
            review_count, average_stars = users[users_id]
            return [(users_id, businesses_id), review_count, average_stars, review_stars, review_counts,float(RestaurantsPriceRange2) if RestaurantsPriceRange2 is not None else None]

start_time = time.time()
#############collect data
yelp_train = train_path + '/yelp_train.csv'
train_dt = before_merge(yelp_train)
test_dt = before_merge(val_path)
business_dt = train_path + '/business.json'
user_dt = train_path + '/user.json'
businesses, users = more_data(business_dt, user_dt)
businesses_dict = dict(businesses.collect())
users_dict = dict(users.collect())
#=================
Train_final = train_dt.map(lambda a: merge_train_data(a, businesses_dict, users_dict))
Test_final = test_dt.map(lambda a: merge_test_data(a, businesses_dict, users_dict))
#=================
# Extract 2 to 6 from Train_final
XTrain = Train_final.map(lambda a: a[2:7]).collect()  #XTrain = [row[2:7] for row in Train_final] 
# Extract last one from Train_final
YTrain = Train_final.map(lambda a: a[-1]).collect() #YTrain = [row[-1] for row in Train_final]
#=================
Xtest = Test_final.map(lambda a: a[1:7]).collect()
Ytest = Test_final.map(lambda a: a[0]).collect()
#=================
X_train, Y_train = np.array(XTrain), np.array(YTrain)
X_test, Y_test = np.array(Xtest), np.array(Ytest)
#print('make sure train',X_train.shape[1])
#print('make sure test',X_test.shape[1])
XGBmodel = xgb.XGBRegressor()
XGBmodel.fit(X_train, Y_train)
prediction = XGBmodel.predict(X_test)
XGB_prediction = np.concatenate((Y_test, prediction.reshape(-1, 1)), axis=1)

with open(outputFile_path,'w', newline= '') as outfile:
    dt_as_csv = csv.writer(outfile)
    dt_as_csv.writerow(['business_id_1', 'business_id_2', 'similarity'])
    for item in XGB_prediction:
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

#23/03/24 23:56:38 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
#“Duration: 34.10055923461914                                                    
#RMSE: 0.9832835202098279


