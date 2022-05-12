import numpy as np
import rrcf
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import os 

data_path_list = ['data/Media/Media_INFO.csv', 'data/Media/Media_LOGIN.csv', 'data/Media/Media_MENU.csv', 'data/Media/Media_STREAM.csv']
# Generate data
df_list = []
for p in data_path_list:
    data = pd.read_csv('./' + p)
    df_list.append(data)

idx_half = df_list[0].index[df_list[0]['Timestamp'] == '20171231_2355-0000'].tolist()[0]
handworkdf = {}
handworkdf['Timestamp'] = df_list[0]['Timestamp']
#handworkdf['INFO_Success'] = df_list[0]['INFO-01-Success']
handworkdf['INFO_Request'] = df_list[0]['INFO-01-Request']

for j in range(1, 2):
    handworkdf['Timestamp'] = df_list[0]['Timestamp']
    #handworkdf['LOGIN_Success_' + str(j)] = df_list[1]['LOGIN-0'+str(j)+'-Success']
    handworkdf['LOGIN_Request_' + str(j)] = df_list[1]['LOGIN-0'+str(j)+'-Request']
    handworkdf['LOGIN_Fail_' + str(j)] = df_list[1]['LOGIN-0'+str(j)+'-Fail']

for j in range(1, 2):
    handworkdf['Timestamp'] = df_list[0]['Timestamp']
    #handworkdf['MENU_Success_' + str(j)] = df_list[2]['MENU-0'+str(j)+'-Success']
    handworkdf['MENU_Request_' + str(j)] = df_list[2]['MENU-0'+str(j)+'-Request']
    handworkdf['MENU_Fail_' + str(j)] = df_list[2]['MENU-0'+str(j)+'-Fail']

for j in range(1, 4):
    handworkdf['STREAM_Session_' + str(j)] = df_list[3]['STREAM-0'+str(j)+'-Session']

if not os.path.isdir('./figure'):
    os.mkdir('./figure')

for key in handworkdf.keys():
    if key != 'Timestamp':
        print(key)
        d = np.array(handworkdf[key])
        d = d[idx_half + 1:]
        d = d[np.isfinite(d)]
        sin = d
        # Set tree parameters
        num_trees = 40
        shingle_size = 4
        tree_size = 256

        # Create a forest of empty trees
        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)

        # Use the "shingle" generator to create rolling window
        points = rrcf.shingle(sin, size=shingle_size)

        # Create a dict to store anomaly score of each point
        avg_codisp = {}

        # For each shingle...
        for index, point in tqdm(enumerate(points)):
            # For each tree in the forest...
            for tree in forest:
                # If tree is above permitted size...
                if len(tree.leaves) > tree_size:
                    # Drop the oldest point (FIFO)
                    tree.forget_point(index - tree_size)
                # Insert the new point into the tree
                tree.insert_point(point, index=index)
                # Compute codisp on the new point...
                new_codisp = tree.codisp(index)
                # And take the average over all trees
                if not index in avg_codisp:
                    avg_codisp[index] = 0
                avg_codisp[index] += new_codisp / num_trees
        
        
        y = sin
        x = np.array(list(range(0, len(y))))
        plt.plot(x, y)

        y = avg_codisp.values()
        score = pd.DataFrame(y, columns=['score'])
        score.to_csv('./figure/score_' +str(num_trees) + '_' + str(shingle_size) + '_' + key + '.csv')
        x = np.array(list(range(0, len(y))))
        
        plt.plot(x, y, 'r')

        plt.savefig('./figure/total_2' + key + '.png')
        plt.clf()

        plt.plot(x, y, 'r')
        plt.savefig('./figure/score_2' + key + '.png')
        plt.clf()