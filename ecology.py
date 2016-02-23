import numpy as np
import pandas as pd
import pickle

import copy
import theano
import theano.tensor as T

import lasagne
import sys
import os
from math import exp,sqrt,sin,cos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

def leakyReLU(x):
	return T.maximum(x,0.08*x)

seed = 12345

np.random.seed(seed)

def float32(k):
    return np.cast['float32'](k)

# Load everything in
dataframe = pd.read_csv("train.csv")

LE = LabelEncoder()
DV = DictVectorizer()
OHE = OneHotEncoder()

# Some of these columns are text, so we want to one-hot-encode them. Columns like ticket number, passenger name, etc are kind of useless. Cabin is a special case - I'm excluding it because it has almost as many values as we have rows of data
cols = [ "Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked" ]
data = np.log(2+DV.fit_transform(dataframe[cols].T.to_dict().values()).todense())

# Now lets add back in the useful numerical features: ticket fare and ages
fares = np.array(dataframe["Fare"]).reshape( (891,1) )

# Get ages and replace empty values with mean
ages = np.array(dataframe["Age"]).reshape( (891,1) )
m_age = np.mean(ages)
ages[ages[:,0]==0] = m_age

# Combine all the data together
data = np.hstack( [data, fares, ages] )

# There were still some missing values in the CSV, so replace them with 0
data[ np.isnan(data) ] = 0

# We end up with 12 variables including all the one-hot-encoded stuff
VARS = data.shape[1]

# Subtract the means and divide by the standard deviations - good for making the neural networks train efficiently
SS = StandardScaler()
data = SS.fit_transform(data)

# Shuffle the data around in case there's any kind of ordering bias
shuf = np.random.permutation(data.shape[0])
data = data[shuf,:]
data = data.astype(np.float32)

# We use 500 rows to train the networks and 392 rows to evaluate the fitness
TRAINCOUNT = 500
TESTCOUNT = 392

# This function will create a neural network with a single hidden layer given a template (input variables and output variable). The output has no nonlinearity, so this is doing regression.
def createNetwork(template):
	net = {}
	
	net["input"] = lasagne.layers.InputLayer(shape=(None,len(template["inputs"])))
	net["dense"] = lasagne.layers.DenseLayer(incoming = net["input"], num_units = template["hidden"], nonlinearity = leakyReLU)
	net["output"] = lasagne.layers.DenseLayer(incoming = net["dense"], num_units = 1, nonlinearity = None)
	
	net["input_buffer"] = T.matrix('inputs')
	net["target_buffer"] = T.matrix('targets')
	
	net["params"] = lasagne.layers.get_all_params( net["output"], trainable = True)
	net["get_output"] = lasagne.layers.get_output(net["output"], net["input_buffer"], deterministic = True)
	net["loss"] = lasagne.objectives.squared_error(net["get_output"], net["target_buffer"]).mean()
	
	net["updates"] = lasagne.updates.adam(net["loss"], net["params"], learning_rate = 1e-1)
	net["train"] = theano.function([net["input_buffer"], net["target_buffer"]], net["loss"], updates=net["updates"])
	net["test"] = theano.function([net["input_buffer"], net["target_buffer"]], net["loss"])
	
	return net

# This function creates a random organism template
def createOrganism():
	org = {}
	
	# Inputs stores the indices of variables used to predict
	org["inputs"] = [np.random.randint(VARS), np.random.randint(VARS), np.random.randint(VARS), np.random.randint(VARS), np.random.randint(VARS)]
	# Outputs stores the index of the variable we're predicting
	org["outputs"] = np.random.randint(VARS)
	# Hidden is the number hidden-layer neurons
	org["hidden"] = 5
	
	# These are derived quantities. Fitness is going to be the performance of the network on the data (squared error). Food will be the amount of food the organism has gathered so far
	org["fitness"] = 0
	org["food"] = 0

	# Do not allow self-prediction!
	while org["outputs"] in org["inputs"]:
		org["inputs"].remove(org["outputs"])
	return org
	
# This function performs mutation on the organism template. The randint(20) calls are giving a 5% chance of doing each of these mutations.
def mutateOrganism(org):
	ilen = len(org["inputs"])
	
	if (ilen>1) and (np.random.randint(20) == 0): # Delete input
		del org["inputs"][np.random.randint(ilen)]
		ilen -= 1
		
	if (ilen<40) and (np.random.randint(20) == 0): # Add input
		org["inputs"].append(np.random.randint(VARS))
		while org["outputs"] in org["inputs"]:
			org["inputs"].remove(org["outputs"])
	
	if (ilen>1) and (np.random.randint(20) == 0): # Change output
		org["outputs"] = np.random.randint(VARS)
		while org["outputs"] in org["inputs"]:
			org["inputs"].remove(org["outputs"])
		
	if (np.random.randint(20) == 0): # Change hidden layer
		org["hidden"] += np.random.randint(3)-1
		if (org["hidden"] < 1):
			org["hidden"] = 1
	org["fitness"] = 0

# This function gets the fitness of an organism (1.0/squared error), after 10 iterations of training.
def evalOrganism(org):
	org["fitness"] = 0
	net = createNetwork(org)
	
	for epoch in range(10):
		net["train"]( data[0:TRAINCOUNT, org["inputs"]].astype(np.float32), data[0:TRAINCOUNT, org["outputs"]:(org["outputs"]+1)].astype(np.float32))
	
	org["fitness"] = 1.0/(1e-8 + net["test"](data[TRAINCOUNT:(TRAINCOUNT+TESTCOUNT), org["inputs"]].astype(np.float32), data[TRAINCOUNT:(TRAINCOUNT+TESTCOUNT), org["outputs"]:(org["outputs"]+1)].astype(np.float32))) - 1.0
	if (org["fitness"] < 0):
		org["fitness"] = 0

# Initialize the population and the food reservoir for each feature
population = []
foodSupply = np.zeros(VARS)
totalFit = np.zeros(VARS)
totalCount = np.zeros(VARS)
	
for i in range(100): # Lets start with 100 organisms - this will change as things starve/etc
	org = createOrganism()
	population.append(org)

# Start the features out with some food so things don't immediately starve
foodSupply += 4.0 * np.ones(VARS)

t = 0
while (t<10000):
	foodSupply += 4.0 * np.ones(VARS) # Add some food to all the features every timestep. If you add more food, you get bigger populations. Less food = smaller populations (and faster runs)
	totalFit = np.zeros(VARS)
	totalCount = np.zeros(VARS)
	delFood = np.zeros(VARS)
	fitTest = 0
	
	# Get all the fitnesses. These will tend to vary a lot at first - anything from 0.01 to 50.
	for i in range(len(population)):
		evalOrganism(population[i])
		totalFit[population[i]["outputs"]] += population[i]["fitness"]
		totalCount[population[i]["outputs"]] += 1
	
	# Organisms get to eat from features based on their fitness proportional to the total fitness of all organisms predicting that feature
	# However, organisms don't get to eat if they can't afford to pay for all the features they read from to do the prediction
	# So we adjust the amount of food used up as each guy eats, to make sure things even out correctly
	# There's a complicated loss function here - the organism metabolism pays 0.3 units of food into each input variable, and expends an extra 0.05 per hidden unit. So we favor small nets.
	for i in range(len(population)):
		earned = foodSupply[population[i]["outputs"]] * population[i]["fitness"]/( 1.0 + totalFit[population[i]["outputs"]] )
		if (earned + population[i]["food"] > 0.3 * len(population[i]["inputs"]) + 0.05 * population[i]["hidden"]): # Enough to pay its ticket...
			population[i]["food"] += earned - 0.3 * len(population[i]["inputs"]) - 0.05 * population[i]["hidden"]
			delFood[population[i]["outputs"]] -= earned
			delFood[population[i]["inputs"]] += 0.3
		else:
			totalFit[population[i]["outputs"]] -= population[i]["fitness"] # Couldn't participate
	
	# Re-adjust the food supply based on what was actually eaten. Carry over the remainder
	foodSupply += delFood
	
	# Now collect some statistics, and have the guys with >1.0 food replicate. Everyone has a flat 20% chance of dying each round, so you have to replicate at least once every 5 rounds to be stable.
	plen = len(population)
	i = 0
	totalFood = 0
	while (i<plen):
		fitTest += population[i]["fitness"]
		totalFood += population[i]["food"]
		if (population[i]["food"] > 1.0): # Enough food to replicate
			population[i]["food"] -= 1.0
			neworg = copy.deepcopy(population[i])
			mutateOrganism(neworg)
			neworg["food"] = 0
			population.append(neworg)
		if (np.random.randint(5) == 0): # Death
			del population[i]
			i -= 1
			plen -= 1
		i += 1
	
	fitness = np.sum( totalFit/(1e-8 + totalCount) )
	
	# Write out statistics of the run
	f = open("stats.txt","a")
	f.write("%d %d %.6g %.6g %.6g\n" % (t, len(population), fitness, totalFood, fitTest))
	f.close()
	
	# Make a survey of what kinds of nets we have. All nets with the same input/output relationship are gathered together as a single 'species',
	# and we collect the population of each species
	outSurvey = {}
	for i in range(VARS+1):
		outSurvey[i] = {}
	
	adj = np.zeros( (VARS,VARS) )
	
	for i in range(len(population)):
		ins = sorted(population[i]["inputs"])
		
		idx = population[i]["outputs"]
		netname = ""
		for j in range(len(ins)):
			adj[idx, ins[j]] += 1 # Here we're assembling statistics for an adjacency matrix. Row index is target variable, column index is input feature, just count up how many things use the column to predict the row
			netname += str(ins[j])
		
		if (netname not in outSurvey[idx]):
			outSurvey[idx][netname] = {}
			outSurvey[idx][netname]["list"] = ins
			outSurvey[idx][netname]["count"] = 0
		
		outSurvey[idx][netname]["count"] += 1		
	
	# Write out the survey. First column is species count, second column is target variable, and the remaining columns are the inputs used.
	f = open("popsurvey/%.6d.txt" % t, "wb")
	for i in range(VARS+1):
		for key in outSurvey[i]:
			f.write("%d %d" % (outSurvey[i][key]["count"], i))
			for j in range(len(outSurvey[i][key]["list"])):
				f.write(" %d" % (outSurvey[i][key]["list"][j]))
			f.write("\n")
	f.close()
	
	# Write out a hypergraph of all our features and predictors as a graphviz input file
	# To keep things simple, only show stuff that has at least a population size of 4. Could make this more stringent to get cleaner graphs/only the strongest connections.
	f = open("nets/%.6d.dot" % t,"wb")
	f.write("strict digraph G {\n")
	for i in range(VARS+1):
		for key in outSurvey[i]:
			if outSurvey[i][key]["count"]>=4:
				f.write("\"%s-%d\" [shape=\"point\"];\n" % (key,i))
				f.write("\"%s-%d\" -> %d;\n" % (key, i, i))
				for j in range(len(outSurvey[i][key]["list"])):
					f.write("%d -> \"%s-%d\";\n" % (outSurvey[i][key]["list"][j], key, i))
	f.write("}\n")
	f.close()
	
	# Save the adjacency matrix of the features.
	np.savetxt("adj/%.6d.txt" % t, adj)
	
	t += 1
