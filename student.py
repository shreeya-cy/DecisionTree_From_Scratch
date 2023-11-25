import argparse
import numpy as np
import pandas as pd

c = 4

def total_entropy(dataset):
    total_rows = dataset.shape[0]
    label = len(dataset.columns)-1
    class_list = dataset[label].unique()
    total_entropy = 0
    for i in class_list:
        label_count = dataset[dataset[label]==i].shape[0]
        etrp = - (label_count/total_rows) * (np.log(label_count/total_rows)/np.log(c))
        total_entropy += etrp
    print("0,root,{},no_leaf".format(total_entropy))

def info_gain(dataset):
    att_no = len(dataset.columns)-1
    label = att_no
    class_list = dataset[label].unique()
    entropy_list = [0]*att_no
    total_count = dataset.shape[0]
    for i in range(0,att_no):
        att_values = dataset[i].unique()
        gain_entropy = 0
        for val in att_values:
            val_dataset = dataset[dataset[i]==val]
            val_count = val_dataset.shape[0]
            total_entropy = 0
            for j in class_list:
                val_label_count = val_dataset[val_dataset[label]==j].shape[0]
                if(val_label_count==0):
                    etrp = 0
                else:
                    etrp = - (val_label_count/val_count) * (np.log(val_label_count/val_count)/np.log(c))
                total_entropy += etrp
            gain_entropy += (total_entropy * (val_count/total_count))
        entropy_list[i] = gain_entropy
    high_info = min(entropy_list)
    return(entropy_list.index(high_info))

def feature_entropies(dataset, attr , depth):
    att_list = dataset[attr].unique()
    label = len(dataset.columns)-1

    pure_class_values = list()
    for i in att_list:
        att_value_data = dataset[dataset[attr]==i]
        att_value_total = att_value_data.shape[0]
        class_list = att_value_data[label].unique()
        total_entropy = 0
        for j in class_list:
            att_value_count = att_value_data[att_value_data[label]==j].shape[0]
            if(att_value_count == len(att_value_data)):
                entropy = 0
                pure_class_values.append(i)
            else:
                entropy = - (att_value_count/att_value_total)*(np.log(att_value_count/att_value_total)/np.log(c))
            total_entropy = total_entropy + entropy
            if(i in pure_class_values):
                clss = j
            else:
                clss = "no_leaf"
        print("{},att{}={},{},{}".format(depth,attr,i,total_entropy,clss))
    return(pure_class_values)


def id3(dataset,attribute_value_pairs,depth):
    if attribute_value_pairs is None:
        attribute_value_pairs = list()
        x = dataset.iloc[:, :-1]
        for col in x.columns:
            val = x[col].unique()
            for v in val:
                attribute_value_pairs.append([col, v])
    if len(attribute_value_pairs)==0:
        return

    attr = info_gain(dataset)
    pure_values = feature_entropies(dataset, attr, depth)
    filtered_attribute_value_pairs = list()
    for i,j in attribute_value_pairs:
        if(i==attr) and (j in pure_values):
            pass
        else:
            filtered_attribute_value_pairs.append([i,j])
    attribute_value_pairs = filtered_attribute_value_pairs
    split = dataset[attr].unique()
    if(len(pure_values)!=0):
        for i in pure_values:
            dataset_filtered = dataset[dataset[attr] != i]
        dataset = dataset_filtered

    for i in split:
        if i not in pure_values:
            sub_dataset = dataset[dataset[attr]==i]
            id3(sub_dataset, attribute_value_pairs, depth = depth+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    dataFile = args.data
    dataset = pd.read_csv(dataFile, header=None)
    label = len(dataset.columns) - 1
    c = dataset[label].nunique()
    total_entropy(dataset)
    attribute_value_pairs = None
    id3(dataset,attribute_value_pairs,1)
