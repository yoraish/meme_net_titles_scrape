# this script is used to convert the json object all_scrape_data.json to x_train x_test, y_train... etc
# computing correct classes scores and dividing data to two groups

import json
from math import log
import matplotlib.pyplot as plt
import numpy as np
import os

def vis(dict, heading):
    plt.figure(0)
    lst = sorted([i for i in dict.values()])

    plotting_bins=np.arange(-2,30,1) # need to add an extra bin when plotting 
    plt.hist(lst, bins=plotting_bins)
    plt.title(heading)
    plt.show()

def generate_groups(all_scrape_data,word_to_ix):
    print("OVERALL NUMBER OF TITLES=", len(all_scrape_data))
    # will go through the data, and take every 9'th one to the test group
    # will also change the number of classes to four
    x_train_scrape = []
    y_train_scrape = []
    x_test_scrape = []
    y_test_scrape = []

    test_data = {}
    train_data = {}
    counter = 0

    for freq_lst, score in all_scrape_data.items():
        if score != 0:
            score = round(log(score,12))-1

            if score > 1:
                score = 1
            if score < 0:
                score = 0
            
        if counter % 9 == 0:
            test_data[freq_lst] = score
            # x_test_scrape.append(freq_lst)
            # y_test_scrape.append(score)
        else:
            train_data[freq_lst] = score
            # x_train_scrape[freq_lst] = score
        counter+=1

    vis(train_data, 'train')
    vis(test_data,'test')

    # save to json files
    with open('train_db.json', 'w') as json_file:  
        json.dump(train_data, json_file)

    with open('test_db.json', 'w') as json_file:  
        json.dump(test_data, json_file)


if __name__ == "__main__":
    with open("word_to_ix.json") as word_to_ix_db:
        # create a dict to map ids to class
        word_to_ix = json.load(word_to_ix_db)

    with open("all_scrape_data.json") as all_scrape_data_db:
        # create a dict to map ids to class
        all_scrape_data = json.load(all_scrape_data_db)
    
    print('length of the all_scrape_data dict = ', len(all_scrape_data))
    generate_groups(all_scrape_data, word_to_ix)