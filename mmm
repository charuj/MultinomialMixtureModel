import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
from collections import Counter

# load the data
import csv
with open ('train_data.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')

    training=[]
    for row in datareader:
        training.append(row)



with open ('train_user_ids.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')

    user_id=[]
    for row in datareader:
        user_id.append(row)


with open ('demographics.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')

    demog=[]
    for row in datareader:
        demog.append(row)

#turning training data into array
training= np.array(training)
training=training.astype(np.float64)
#training=np.random.permutation(training)

#3-fold cross validation, dividing the data
N=training.shape[0]
set1= training[0:N/3]
set2= training[N/3:2*N/3]
set3=training[2*N/3:N]

N=set1.shape[0]
#E Step: calculate responsbility
def beta(K):

    betas= np.random.random([89, K,5])
    for i in range(0,89):#put variable for 89
        for j in range(0,K):
            beta_sum=np.sum(betas[i,j,:])
            beta_normalized= betas[i,j,:]/beta_sum
            betas[i,j,:] = beta_normalized
    return betas, np.ones(K)*(1.0/K)


def estep(K, beta, data, N, priors):
    a=np.zeros([K,N])
    responsibility= np.zeros([K,N])
    for users in range(0, N): # users
        for clusters in range(0,K):
            likelihood=1
            for movies  in range(0,89): #movies
                x= data[users, movies] #rating for a particular user for a particular movie
                x=int(x)
                if x == 0:
                    continue
                likelihood= beta[movies, clusters,x-1]*likelihood #likelihood for particular user, for particular movie, for particular cluster
                #prior=np.float64(1.0/K)

            pre_a= priors[clusters] * likelihood # not summed yet
            # sum over the 89 movies now
            # if pre_a == 0:
            #     print "a is zero"
            try:
                a[clusters,users]+= pre_a
            except:
                print 'alsdfasd'
    for users in range(0,N):
        # calc scaler log sum exp for each user
        #sum_user= sc.misc.logsumexp(a[:, users])
        sum_user= np.sum(a[:,users])
        # if np.min(a[:,users]) == 0:
        #     print "ASASf"
        # if np.min(a[:,users])/(sum_user) == 0:
        #     print "respons 0"
        #responsibility[:,users]= np.exp(np.log(a[:,users])-(sum_user+1*np.e**(-32)))
        responsibility[:,users]= (a[:,users])/(sum_user)

    return responsibility, a

def mstep(responsibility,data,K,priors):
    betas= np.zeros([89, K,5])
    for movie in range(0,89):#put variable for 89
        for cluster in range(0,K):
            rating_count= [0,0,0,0,0]
            for users in range(0,N):
                if data[users, movie]!= 0:
                    #TODO:the following might be for "soft"
                 rating_count[int(data[users,movie])-1]+=responsibility[cluster,users]
                    #hard version
                 #rating_count[int(data[users,movie])-1]+=1.0

                 # normalize
            beta_ml= rating_count/(np.sum(rating_count)) # for one particular cluster movie combo
            beta_ml=beta_ml +1*np.e**(-32)
            beta_ml=beta_ml/(np.sum(beta_ml))
            betas[movie, cluster,:]= beta_ml
            #TODO: WRITE CODE TO UPDATE PRIORS
    distribution_across_clusters = cluster_dist(K,responsibility)
    distribution_across_clusters=np.array(distribution_across_clusters)
    priors = distribution_across_clusters.astype(np.float64) / np.sum(distribution_across_clusters)
    return betas,priors


def log_like(a, responsibility):
    # max_prob= np.max(a, axis=0) # max a across clusters
    # log_maxprob= np.log(max_prob)
    # log_probx= np.sum(log_maxprob)
    log_probx= np.multiply(responsibility, np.log(a + (1*np.e**(-32))))
    log_probx= np.sum(log_probx)
    return log_probx

def demographics(N,K,a, userid, demog):
    userid = np.array(userid[0]).astype(int)
    cluster_users_belong_to= np.argmax(a, axis=0) # max a across clusters
    list_of_cluster_userids= []
    list_of_cluster_ages = []
    list_of_cluster_genders = []

    for i in range(K):
        list_of_cluster_userids.append([])
        list_of_cluster_ages.append([])
        list_of_cluster_genders.append([])

    #for loop i to range of 241
    for i in range (N):
        cluster = cluster_users_belong_to[i]
        list_of_cluster_userids[cluster].append(userid[i])


    for cluster in range (K):
        user_ids_in_this_cluster =  list_of_cluster_userids[cluster]
        for user_id_in_cluster in user_ids_in_this_cluster:
            age = int(demog[user_id_in_cluster][1])
            gender = demog[user_id_in_cluster][2]
            list_of_cluster_ages[cluster].append(age)
            list_of_cluster_genders[cluster].append(gender)

        print "mean age for cluster ",cluster," is ", np.mean(np.array(list_of_cluster_ages[cluster]))
        males = 0
        females = 0
        for gender in list_of_cluster_genders[cluster]:
            if gender == 'M':
                males +=1
            else:
                females+=1
        print "males ", males, " females ", females, " ratio ", float(males)/females


'''def cluster_dist(responsibility):
    best_cluster= np.argmax(responsibility, axis=0) # finding maximal cluster, for every user
    dict= Counter(best_cluster).values()

    #dict= dict.T
    print dict
    #plt.hist(dict)
    #plt.show()
    return dict
    '''
def cluster_dist(K, responsibility):
    dict= np.zeros(K)
    best_cluster= np.argmax(responsibility, axis=0)
    for i in range(0,N):

        dict[best_cluster[i]]+=1


        #plt.hist(count)
        #plt.show(count)

    return dict

list_logk= []
list_logk2=[]
for restart in range(0,1):


    # main

    # need to iterate over different values of k, and do cross-validation
    #K=[1,2,3,4,5,6,7,8,9,10]
    K=[1,2,3,4]
    log_by_K = []
    log_by_K2 = []
    epochs = 10
    t=np.arange(0,epochs)


    for num in K:
        sum_ll= 0
        sum_ll2= 0
        list_results=[]
        list_results2=[]
        for crossv in range(1):
            all_sets= [set1, set2, set3]
            valid_set=all_sets.pop(crossv)
            train_set= np.concatenate(all_sets)

            print "\n" +"for K value = " + str(num)
            betas,priors= beta(num)
            results = []
            results2=[]

            for i in range(0,epochs):
                #set1= np.random.permutation(set1)
                responsibility, a= estep(num,betas,train_set, N,priors)
                responsibility2, a2= estep(num,betas,valid_set, N,priors)
                log_probx= log_like(a, responsibility)
                log_probx2= log_like(a2, responsibility2)
                dist= cluster_dist(num, responsibility)
                #print log_probx


                results.append(log_probx)
                results2.append(log_probx2)



                betas,priors= mstep(responsibility,train_set,num,priors)


            list_results.append(results)
            list_results2.append(results2)


            sum_ll= sum_ll+ log_probx
            sum_ll2= sum_ll2+ log_probx2



        # take means of log likelihood over training sets
        log_probx= np.mean(sum_ll)
        log_probx2= np.mean(sum_ll2)

        list_results= np.array(list_results)
        list_results2=np.array(list_results2)
        results_mean= np.mean(list_results, axis=0)
        results2_mean= np.mean(list_results2, axis=0)

        log_by_K.append(log_probx)
        log_by_K2.append(log_probx2)
        demographics(N,num,a, user_id, demog)


        # plt.plot(t,results_mean, 'r--', t, results2_mean, 'b')
        # plt.show()

    list_logk.append(log_by_K)
    list_logk2.append(log_by_K2)
    K=np.array(K)


    #fig,ax= plt.subplots()
    #rects1= ax.bar(K, log_by_K, 0.25, color='r')
    #rects2= ax.bar(K +0.25, log_by_K2, 0.25, color='b')
    #plt.bar(K,log_by_K, K,log_by_K2)
    #plt.show()

# list_logk=np.array(list_logk)
# list_logk2=np.array(list_logk2)
# std_list_logk=np.std(list_logk,axis=0)
# std_list_logk2= np.std(list_logk2,axis=0)
#
# print std_list_logk
# print std_list_logk2
# print np.mean(list_logk,axis=0)
# print np.mean(list_logk2,axis=0)








