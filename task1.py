from pyspark import SparkContext, SparkConf
import csv
import sys
import time
from itertools import combinations
import random

#https://towardsdatascience.com/locality-sensitive-hashing-how-to-find-similar-items-in-a-large-set-with-precision-d907c52b05fc
#https://zhuanlan.zhihu.com/p/46164294

inputFile_path = sys.argv[1] #yelp_train.csv
outputFile_path = sys.argv[2]

conf = SparkConf()
conf.setMaster("local[*]").setAppName("hw3-task1") #set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
#Use SparkContext library and read file as text and then map it to json.
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

start_time = time.time()

dtRDD = sc.textFile(inputFile_path)
header = dtRDD.first()
dtRDD_0 = dtRDD.filter(lambda a: a != header)
dtRDD_real = dtRDD_0.map(lambda x: (x.split(',')))

#find out all the unique users in the yelp_train.csv:
dtRDD_final = dtRDD_real.map(lambda x: x[0]).distinct().collect() #sortBy(lambda x: x[0])

total_users = len(dtRDD_final)
#need to create businesses_dict
users_dict = dict(zip(dtRDD_final, range(total_users)))
#print('\nuser dict:\n',list(users_dict.keys())[:4])
#print('\nuser dict length:\n',len(list(users_dict.keys())))

#In instruction, hash function can be cal like: f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m
#where p is any prime number and m is the number of bins. 

#So, step1: create custom hash function with unique a, b and p(prime number)
#because this way can help to improve the distribution of hash values, improve the performance for LSH

num_hash_functions = 100
#======================== setting the hashing parameter number
def custom_hashfunction(nums, counts):
    res = []
    a = random.sample(range(1,counts), nums)  
    res.append(a)
    #print("random_a: {}\n".format(a[:4]))
    b = random.sample(range(1,counts),nums)
    res.append(b)
    #print("random_b: {}\n".format(b[:4]))
    def generate_primenumber(num, words):
        primes = []
        word = words + 1
        count = len(primes)
        while count < num:
            confirm_prime = True
            for i in range(2, int(word ** 0.5) + 1):
                if word % i == 0:
                    confirm_prime = False
                    break
            if confirm_prime == True:
                primes.append(word)
                count += 1
            word += 1
        return primes
    p = generate_primenumber(nums, counts)
    #print("random_p: {}\n".format(p[:4]))
    res.append(p)
    #print("random_res: {}\n".format(res[:4]))
    return res

h = custom_hashfunction(num_hash_functions,total_users)

#============================= then the character signature matrix
#needs users_dict, num_hash_functions, total_users, res, dataset
businesses_dtRDD = dtRDD_real.map(lambda x: (x[1],users_dict[x[0]])).groupByKey().mapValues(list).map(lambda y: (y[0],sorted(y[1]))).sortBy(lambda x: x[0])
#print("\n data_be4_mx: {}\n".format(businesses_dtRDD.collect()[:3]))

def mini_hash(rows, res_h, counts,nums):
    hash_value = []
    for i in range(len(res_h[0])): #nums
        #f(x) = ((ax + b) % p) % m
        x = nums
        mini_hash_values = float('inf') #2 ** 32 - 1   
        a = res_h[0]
        b = res_h[1]
        p = res_h[2]
        m = counts

        for xcj in rows:
            cal_res = ((a[i] * xcj + b[i]) % p[i]) % m
        #mini_hash_values[i] = min(mini_hash_res, mini_hash_values[i])
            if cal_res < mini_hash_values:
                mini_hash_values = cal_res
                #print(mini_hash_values)
        hash_value.append(mini_hash_values)
    return hash_value

hash_values = businesses_dtRDD.map(lambda x: (x[0], mini_hash(x[1], h, total_users, num_hash_functions)))
#print("signature_mx: {}\n".format(hash_values.collect()[:4]))
#================ split the character matrix
#needs to confirm the individual bands number & rows per bands number
num_bands = 50

def bucket(dt, nums, num_bands):
    band_size = nums // num_bands
    bucket_res = []
    bucket_index, hash_values_list = dt[0], dt[1]
    for i in range(num_bands):
        j = i * band_size
        k = j + band_size
        bands = tuple(hash_values_list[j:k])
        bucket_res.append(((i+1, bands),bucket_index))
    return bucket_res
bucket_values = hash_values.flatMap(lambda x: bucket(x,num_hash_functions,num_bands))
#print("after_bands: {}\n".format(bucket_values.collect()[:4]))
#================== generate the set of candidate pairs
similar_business = bucket_values.groupByKey().mapValues(list).filter(lambda x: len(x[1]) > 1)

def similar_bus(dt):
    similar_bus_res = sorted(dt[1])
    combs_res = combinations(similar_bus_res,2)
    return combs_res

similar_business = similar_business.flatMap(lambda x: combinations(sorted(x[1]),2)).distinct()
#print("after_cand_pair: {}\n".format(similar_business.collect()[:4]))
#================== computing the Jaccard Similarity for the candidate pairs
def jaccard_similarity(set_a, set_b):
    set1 = set(set_b[set_a[0]])
    set2 = set(set_b[set_a[1]])
    intersection = set1.intersection(set_b[set_a[1]])
    union = set1.union(set2)
    lsh_res = len(intersection)/ len(union)
    return (sorted(set_a), lsh_res)

#businesses_dtRDD = dtRDD_real.map(lambda x: (x[1],users_dict[x[0]])).groupByKey().mapValues(list).map(lambda y: (y[0],sorted(y[1]))).sortBy(lambda x: x[0])
comp = businesses_dtRDD.collect()
comp = dict(comp)
jaccard_similarity_res = similar_business.map(lambda x: jaccard_similarity(x, comp))
#print("\njaccard_sim_0: {}\n".format(jaccard_similarity_res.collect()[:3]))
jaccard_similarity_res = jaccard_similarity_res.filter(lambda x: x[1] >= 0.5).sortBy(lambda x: x[0]).collect()
final = []
for tup, simi in jaccard_similarity_res:
    new_tup = tuple(tup)
    final.append((new_tup,simi))

#print('test1', jaccard_similarity_res[0][1])
#======================= xiaocaiji
with open(outputFile_path,'w', newline= '') as outfile:
    dt_as_csv = csv.writer(outfile)
    dt_as_csv.writerow(['business_id_1', 'business_id_2', 'similarity'])
    for item, simi in jaccard_similarity_res:
        dt_as_csv.writerow([item[0],item[1],simi])

end_time = time.time()
exe_time = end_time - start_time
print("“Duration: {}".format(exe_time))