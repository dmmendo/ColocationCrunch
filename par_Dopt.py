
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
from sklearn.preprocessing import normalize

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
from sklearn.model_selection import train_test_split

def D_func(X):
  return np.linalg.det(np.dot(X.T,X))

def MAE(truth,predictions,names,indivRun):
  total = 0
  for i in range(0,len(truth)):
    total = total + abs(truth[i] - predictions[i])*indivRun[names[i][0]]
    #print(truth[i]*indivRun[names[i]] + indivRun[names[i]],names[i])
  return total / len(truth)

"""## Limited Data Experiments"""
print("begin experiment")
num_procs = 50
num_trials = 10000
this_train_sizes = np.linspace(0.05,1,20)
results = Manager().list([0 for i in range(len(this_train_sizes)*num_trials)])
dopt_results = Manager().list([0 for i in range(len(this_train_sizes)*num_trials)])

def uncertainty(reg,profile_features,modlist,indivRuntimes,available_mask):
  tree_predictions = []
  runtime_vec = np.array([indivRuntimes[mod[0]] for mod in modlist])
  #runtime_vec = np.array([1 for mod in modlist])
  for tree in range(100):
      tree_predictions.append(reg.estimators_[tree].predict(profile_features) * runtime_vec * available_mask)
  return np.var(tree_predictions,axis=0)

def run_trial(profile_features,labels,modlist,this_train_sizes,results,dopt_results,n,num_trials):
  print("trial",n)
  random.seed(n)
  np.random.seed(n)
  profile_features,labels,modlist = shuffle(normalize(profile_features,axis=0),labels,modlist)
  for i in range(len(this_train_sizes)):
    dopt_val_max = float('-inf')
    best_X_train = None
    best_y_train = None
    for j in range(num_trials):
      if 1-this_train_sizes[i] > 0:
        cur_X_train, cur_X_test, cur_y_train, cur_y_test = train_test_split(profile_features,labels,test_size=1-this_train_sizes[i])
      else:
        cur_X_train, cur_y_train = profile_features,labels
      tmp_val = D_func(cur_X_train)
      if tmp_val > dopt_val_max:
        dopt_val_max = tmp_val
        best_X_train = cur_X_train
        best_y_train = cur_y_train
    reg = RandomForestRegressor().fit(best_X_train,best_y_train)
    dopt_results[n*len(this_train_sizes) + i] = dopt_val_max
    results[n*len(this_train_sizes) + i] = MAE(labels,reg.predict(profile_features),modlist,indivRuntimes)

procs = []
for n in range(num_procs):
  #run_trial(profile_features,labels,modlist,this_train_sizes,results,n)
  #profile_features,labels,modlist = shuffle(profile_features,labels,modlist)
  proc_num_trials = int(num_trials/num_procs)
  p = Process(target=run_trial, args=(profile_features,labels,modlist,this_train_sizes,results,dopt_results,n,proc_num_trials))
  p.start()
  procs.append(p)
for n in range(num_procs):
  procs[n].join()

results = np.array(results).reshape((num_trials,len(this_train_sizes)))
dopt_results = np.array(dopt_results).reshape((num_trials,len(this_train_sizes)))

dopt_idx = np.argmax(dopt_results,axis=0)
ret_results = np.zeros(len(this_train_sizes))
for i in range(len(this_train_sizes)):
  ret_results[i] = results[dopt_idx[i]][i]
    

json.dump(ret_results.tolist(),open("dopt_10000sim.json","w"))
json.dump(this_train_sizes.tolist(),open("trainsize_dopt_10000sim.json","w"))
