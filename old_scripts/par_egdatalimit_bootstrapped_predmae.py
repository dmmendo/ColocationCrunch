# -*- coding: utf-8 -*-
"""ColocateDataCollection_minpredmae.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LKIStYwlc80w1wdGm5Dn_mgBaWV6S1Z-
"""

import numpy as np
import time
import random
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import json
from multiprocessing import Process,Manager

import time

from itertools import permutations
from itertools import combinations

def process_indivRuntimes(filename):
  f = open(filename)
  indivRuntimes = {}
  for line in f:
    entry = line.split("\n")[0]
    entry = entry.split("\t")
    indivRuntimes[entry[0]] = float(entry[1])
  return indivRuntimes

indivRuntimesFile = "indivRuntimes.txt"
indivRuntimes = process_indivRuntimes(indivRuntimesFile)

#for TPU runtimes:
#for entry in indivRuntimes:
#  indivRuntimes[entry] /= 1000

"""## Data Setup"""

def indivLatencyData(indivRuntimes,feature_dict):
  features = []
  labels = []
  for mod in indivRuntimes:
    if mod in feature_dict:
      features.append(feature_dict[mod])
      labels.append(indivRuntimes[mod])
  features = np.array(features)
  features = features.reshape((features.shape[0],features.shape[1]))
  return features,labels

def process_indiv_data(filename):
  f = open(filename,"r")
  indivDict = {}
  for line in f:
    entry = line.split('\n')[0]
    entry = entry.split('\t')
    indivDict[entry[0]] = [float(entry[i]) for i in range(1,len(entry))]
  return indivDict

def my_train_test_split(profile_features,features,labels, test_size):
  t_size = int(test_size*len(labels))
  return profile_features[t_size:],features[t_size:],profile_features[0:t_size],features[0:t_size],labels[t_size:],labels[0:t_size]

indiv_profile_filename = "V1.txt"

profile_features,labels = indivLatencyData(indivRuntimes,process_indiv_data(indiv_profile_filename))
profile_features,labels = shuffle(profile_features,labels)

#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
#base_X_train, X_train, base_X_test, X_test, y_train, y_test = my_train_test_split(profile_features,features, labels, test_size=0.2)

def process_colocate_data(filename,indivRun):
  f = open(filename,"r")
  colocDict = {}
  for line in f:
    entry = line.split('\n')[0]
    entry = entry.split('\t')
    if entry[0] not in colocDict:
      colocDict[entry[0]] = {entry[1]:(float(entry[2])-float(indivRun[entry[0]]))/float(indivRun[entry[0]])}
    else:
      colocDict[entry[0]][entry[1]] = (float(entry[2])-float(indivRun[entry[0]]))/float(indivRun[entry[0]])
  return colocDict

def get_train_data(colocDict,prof_indivDict):
  y = []
  prof_x = []
  modlist = []
  for mod1 in colocDict:
    for mod2 in colocDict[mod1]:
      y.append(colocDict[mod1][mod2])
      prof_x.append(prof_indivDict[mod1] + prof_indivDict[mod2])
      modlist.append((mod1,mod2))
  y = np.array(y)
  prof_x = np.array(prof_x)
  return prof_x,y,modlist

def process_multicolocate_data(filename,indivRun):
  f = open(filename,"r")
  colocDict = {}
  for line in f:
    entry = line.split('\n')[0]
    entry = entry.split('\t')
    key = []
    for i in range(1,len(entry)-1):
      key.append(entry[i])
    key.sort()
    if entry[0] not in colocDict:
      colocDict[entry[0]] = {tuple(key):(float(entry[-1])-indivRun[entry[0]])/indivRun[entry[0]]}
      #colocDict[entry[0]] = {tuple(key):(float(entry[-1])-indivRun[entry[0]])}
    else:
      colocDict[entry[0]][tuple(key)] = (float(entry[-1]) - indivRun[entry[0]])/indivRun[entry[0]]
      #colocDict[entry[0]][tuple(key)] = (float(entry[-1]) - indivRun[entry[0]])
  return colocDict

def get_train_data_multicolocate(colocDict,indivDict,prof_indivDict):
  y = []
  x = []
  prof_x = []
  multimodlist = []
  for mod1 in colocDict:
    for modlist in colocDict[mod1]:
      for instance_modlist in permutations(modlist,len(modlist)):
        next_ex = indivDict[mod1]
        prof_next_ex = prof_indivDict[mod1]
        for mod in instance_modlist:
          next_ex = np.concatenate((next_ex,indivDict[mod]))
          prof_next_ex = prof_next_ex + prof_indivDict[mod]
        y.append(colocDict[mod1][modlist])
        x.append(next_ex)
        prof_x.append(np.array(prof_next_ex))
        multimodlist.append((mod1,modlist))
  x = np.array(x)
  x = x.reshape((x.shape[0],x.shape[1]))
  y = np.array(y)
  prof_x = np.array(prof_x)
  print(x.shape)
  return np.array(prof_x),np.array(x),np.array(y),multimodlist

def get_train_partition(data,split):
  return data[0:int(split*len(data))]
def get_test_partition(data,split):
  return data[int(split*len(data)):len(data)]

colocDict = process_colocate_data("colocRuntimes.txt",indivRuntimes)
profile_features,labels,modlist = get_train_data(colocDict,process_indiv_data(indiv_profile_filename))

from sklearn.ensemble import RandomForestRegressor

def MAE(truth,predictions,names,indivRun):
  total = 0
  for i in range(0,len(truth)):
    total = total + abs(truth[i] - predictions[i])*indivRun[names[i][0]]
    #print(truth[i]*indivRun[names[i]] + indivRun[names[i]],names[i])
  return total / len(truth)

"""## Limited Data Experiments"""
print("begin experiment")
num_trials = 16
this_train_sizes = np.linspace(1/len(labels),1,len(labels))
results = [0 for i in range(len(this_train_sizes)*num_trials)]
results = Manager().list([0 for i in range(len(this_train_sizes)*num_trials)])

def uncertainty(reg,profile_features,modlist,indivRuntimes,available_mask):
  tree_predictions = []
  runtime_vec = np.array([indivRuntimes[mod[0]] for mod in modlist])
  #runtime_vec = np.array([1 for mod in modlist])
  for tree in range(100):
      tree_predictions.append(reg.estimators_[tree].predict(profile_features) * runtime_vec * available_mask)
  return np.var(tree_predictions,axis=0)

def gen_eval_set(profile_features,labels,modlist,eval_set,n_ex):
  epsilon = 0.3
  if np.random.binomial(1,epsilon) == 1:
    eval_idx = random.sample([i for i in range(len(labels))],n_ex)
    for idx in eval_idx:
      eval_set.add(idx)
  this_eval_set = random.sample(list(eval_set),int(0.1*len(labels)))
  return np.array([profile_features[idx] for idx in this_eval_set]),np.array([labels[idx] for idx in this_eval_set]),np.array([modlist[idx] for idx in this_eval_set]),eval_set

def run_trial(profile_features,labels,modlist,this_train_sizes,results,n):
  print("trial",n)
  random.seed(n)
  np.random.seed(n)
  profile_features,labels,modlist = shuffle(profile_features,labels,modlist)
  cur_X_train,cur_y_train = profile_features[0:int(np.ceil(len(labels)*this_train_sizes[0]))],labels[0:int(np.ceil(len(labels)*this_train_sizes[0]))]
  available_sample = set([i for i in range(int(np.ceil(len(labels)*this_train_sizes[0])),len(labels))])
  cur_reg = RandomForestRegressor().fit(cur_X_train,cur_y_train)
  results[n*len(this_train_sizes)] += MAE(labels,cur_reg.predict(profile_features),modlist,indivRuntimes)
  eval_set = set(random.sample([i for i in range(len(labels))],int(0.1*len(labels))))
  for i in range(1,len(this_train_sizes)):
    samples = []
    sample_costs = []
    num_to_profile = max(1,int(np.floor(len(labels)*(this_train_sizes[i]-this_train_sizes[i-1]))))
    available_list = shuffle(list(available_sample)) #for num_to_profile = 1
    eval_sample_size = 1
    eval_features,eval_labels,eval_modlist,eval_set = gen_eval_set(profile_features,labels,modlist,eval_set,eval_sample_size)
    K = random.randint(0,99)
    for j in range(len(available_sample)):
      if num_to_profile == 1:
        new_sample_idx = [available_list[j]]
      else:
        new_sample_idx = random.sample(available_sample,min(num_to_profile,len(available_sample)))
      new_X_train = np.array([profile_features[idx] for idx in new_sample_idx])
      new_y_train = np.array([labels[idx] for idx in new_sample_idx])
      reg = RandomForestRegressor().fit(np.concatenate((cur_X_train,new_X_train)),np.concatenate((cur_y_train,cur_reg.estimators_[K].predict(new_X_train))))
      #reg = RandomForestRegressor().fit(np.concatenate((cur_X_train,new_X_train)),np.concatenate((cur_y_train,new_y_train)))
      samples.append((new_X_train,new_y_train,new_sample_idx))
      sample_costs.append(MAE(eval_labels,reg.predict(eval_features),eval_modlist,indivRuntimes))
      
    best_sample_idx = np.argmin(sample_costs)
    available_sample = available_sample - set(samples[best_sample_idx][2])
    cur_X_train = np.concatenate((cur_X_train,samples[best_sample_idx][0]))
    cur_y_train = np.concatenate((cur_y_train,samples[best_sample_idx][1]))
    cur_reg = RandomForestRegressor().fit(cur_X_train,cur_y_train)
    results[n*len(this_train_sizes) + i] += MAE(labels,cur_reg.predict(profile_features),modlist,indivRuntimes)

procs = []
for n in range(num_trials):
  #run_trial(profile_features,labels,modlist,this_train_sizes,results,n)
  #profile_features,labels,modlist = shuffle(profile_features,labels,modlist)
  p = Process(target=run_trial, args=(profile_features,labels,modlist,this_train_sizes,results,n))
  p.start()
  procs.append(p)
for n in range(num_trials):
  procs[n].join()

results = np.array(results).reshape((num_trials,len(this_train_sizes)))
results = np.sum(results,axis=0)
results /=num_trials

json.dump(results.tolist(),open("egdatalimit_boostrapped_predmae_10sim.json","w"))
json.dump(this_train_sizes.tolist(),open("trainsize_egdatalimit_bootstrapped_predmae_10sim.json","w"))