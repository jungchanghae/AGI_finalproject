import random
import json

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

def create_template(class_labels, gold_label, candidate_num):
    candidate_class = {gold_label}
    while (len(candidate_class)) < candidate_num:
        candidate_class.add(random.choice(list(class_labels)))
    candidate_class = list(candidate_class)

    random.shuffle(candidate_class)
    gold_index = candidate_class.index(gold_label)
    # template = (f"Choose your answer between "
    #             # f"1={candidate_class[0]}, 2={candidate_class[1]}, 3={candidate_class[2]}, 4={candidate_class[3]}"
    #             # f". Answer the question with integer form."
    #             f"'{candidate_class[0]}', '{candidate_class[1]}', '{candidate_class[2]}'"
    #             f", '{candidate_class[3]}' that best describes the contexts."
    #             )
    template = "Choose your answer between "
    for i in range(candidate_num-1):
        template += f"'{candidate_class[i]}', "
    template += f"'{candidate_class[-1]}' that best describes the contexts."
    
    return template, gold_index + 1


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