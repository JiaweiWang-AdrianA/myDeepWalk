from tqdm import tqdm
from collections import defaultdict


def getDatasetFromFile(filePath:str):
	""" get dataset from file
		input: file path (str)
		output: dataset (list) 
				- dataset's element is a tuple
	"""
	dataset = []
	f = open(filePath, 'r')
	lines = f.readlines()
	print('read dataset: ')
	for line in tqdm(lines):
		line = line.strip('\n')
		datas = line.split(' ')
		dataset.append(tuple(datas))
	f.close()
	return dataset


def getSimplestGraph(dataset:list, saveFilePath='./dataset/Simplest_Graph.txt'):
	""" generate weight_graph (only considering the communication) from dataset
		input: dataset(list), saveFilePath (str)
		output: adjacency list of simplest_graph (dict)
						- key (str): id_x; 
						- values (set): ids (idx's adjacent node)
	"""
	simplest_graph = defaultdict(set)
	# get simplest_graph
	print('get gragh from dataset: ')
	for (time, fromId, toId) in tqdm(dataset):
		if toId == fromId: # sender == receiver
			continue
		simplest_graph[fromId].add(toId)
	# save the simplest_graph to file
	print('write gragh to the file: ')
	save_str = ''
	for (fromId, toIds) in tqdm(simplest_graph.items()):
		for toId in toIds:
			save_str += str(fromId) + ' ' + str(toId) + ' ' + '\n'
	f = open(saveFilePath, 'w')
	f.write(save_str)
	f.close()
	# # print the dataset and the graph
	# print('dataset: ')
	# for (time, fromId, toId) in dataset:
	# 	print((fromId, toId))
	# print('weight_graph: ')
	# f = open(saveFilePath, 'r')
	# for line in f.readlines():
	# 	print(line, end='')
	# f.close()
	return simplest_graph, saveFilePath


def getWeightGraph(dataset:list, saveFilePath='./dataset/Weight_Graph.txt'):
	""" generate weight_graph (only considering the communication) from dataset
		input: dataset(list), saveFilePath (str)
		output: adjacency list of weight_graph (dict)
						- key (str): id_x; 
						- values (dict): 
							key (str): id_y (idx's adjacent node)
							value (int): the weight(communication times) of edge(id_x, id_y)
	"""
	weight_graph = defaultdict(dict)
	# get weight_graph
	print('get gragh from dataset: ')
	for (time, fromId, toId) in tqdm(dataset):
		if toId == fromId: # sender == receiver
			continue
		if toId in weight_graph[fromId]:
			weight_graph[fromId][toId] += 1
		else:
			weight_graph[fromId][toId] = 1
	# save the weight_graph to file
	print('write gragh to the file: ')
	save_str = ''
	for (fromId, toIds_dict) in tqdm(weight_graph.items()):
		for (toId, weight) in toIds_dict.items():
			save_str += str(fromId) + ' ' + str(toId) + ' ' + str(weight) + '\n'
	f = open(saveFilePath, 'w')
	f.write(save_str)
	f.close()
	print('writing done. The walks is saved in [' + saveFilePath +']')
	# # print the dataset and the graph
	# print('dataset: ')
	# for (time, fromId, toId) in dataset:
	# 	print((fromId, toId))
	# print('weight_graph: ')
	# f = open(saveFilePath, 'r')
	# for line in f.readlines():
	# 	print(line, end='')
	# f.close()
	return weight_graph, saveFilePath




# dataset = getDatasetFromFile('./dataset/Enron_TimeFromTo.txt')
# print(len(dataset))
# getGraphWithoutTime('./dataset/Enron_TimeFromTo.txt')