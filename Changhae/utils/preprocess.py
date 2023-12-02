import random
import json
import collections
import re

def load_class_label(dir):
    # food101 json file
    with open(dir, "r") as f:
        data = json.load(f)
    return data['default']['features']['label']['names']

def get_class_label(df):
    class_set = set()
    # num_set = set()
    for row in df:
        class_set.add(row['english_label'][0])
    return class_set

def create_template(class_labels, gold_label, candidate_num, template_idx=0):
    candidate_class = {gold_label}
    while (len(candidate_class)) < candidate_num:
        candidate_class.add(random.choice(list(class_labels)))
    candidate_class = list(candidate_class)

    random.shuffle(candidate_class)
    gold_index = candidate_class.index(gold_label)
    
    if template_idx == 0:
        template = "Choose your answer between "
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i]}', "
        template += f"'{candidate_class[-1]}' that best describes the contexts."
    elif template_idx == 1:
        template = "Choose your answer between ["
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i]}', "
        template += f"'{candidate_class[-1]}'] that best describes the contexts."
    elif template_idx == 2:
        template = 'Please choose from samples of what this food is. Sample : '
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i]}', "
        template += f"'{candidate_class[-1]}'."
    elif template_idx == 3:
        template = 'Please choose from samples of what this food is. Sample : [ '
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i]}', "
        template += f"'{candidate_class[-1]}' ]."
        
    return template, gold_index

def most_common_word(paragraph, class_list, ban_list, candidate_num, template_idx):
    pull_class_list = [ k for c in class_list for k in c.split('_')] 
    words = [word for word in re.sub(r'[^\w]',' ', paragraph).lower().split() if word in class_list+pull_class_list if word not in ban_list]
    counts = collections.Counter(words)
    candidate_class = []
    if len(counts.most_common(1)) != 0:
        for i in class_list:
            if counts.most_common(1)[0][0] in i:
                candidate_class.append(i)
            if (len(candidate_class)) == candidate_num:
                break
    while (len(candidate_class)) < candidate_num:
        candidate_class.append(random.choice(list(class_list)))
    
    if template_idx == 0:
        template = "What is this food name? Choose your answer between "
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i].lower().replace('_', ' ')}', "
        template += f"'{candidate_class[-1].lower().replace('_', ' ')}' that best describes the contexts."
    elif template_idx == 1:
        template = "What is this food name? Choose your answer between ["
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i].lower().replace('_', ' ')}', "
        template += f"'{candidate_class[-1].lower().replace('_', ' ')}'] that best describes the contexts."
    elif template_idx == 2:
        template = 'What is this food name? Please choose from samples of what this food is. Sample : '
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i].lower().replace('_', ' ')}', "
        template += f"'{candidate_class[-1].lower().replace('_', ' ')}'."
    elif template_idx == 3:
        template = 'What is this food name? Please choose from samples of what this food is. Sample : [ '
        for i in range(candidate_num-1):
            template += f"'{candidate_class[i].lower().replace('_', ' ')}', "
        template += f"'{candidate_class[-1].lower().replace('_', ' ')}' ]."
    return template

def evaluate_answer(pred_label, gold_label, matched_num, sub_matched_result):
    # Exact Accuracy
    gold_label = gold_label.lower().replace("_", " ")
    pred_label = pred_label.replace("_", " ")
    # print(gold_label, pred_label)
    if gold_label in pred_label:
        matched_num += 1

    return matched_num, sub_matched_result

def label_int2str(class_labels, label_int):
    
    return class_labels[label_int]