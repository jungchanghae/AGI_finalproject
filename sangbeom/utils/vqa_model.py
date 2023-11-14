import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from lavis.common.gradcam import getAttMap

def postprocess_Answer(text):
    for i, ans in enumerate(text):
        for j, w in enumerate(ans):
            if w == '.' or w == '\n':
                ans = ans[:j].lower()
                break
    return ans


def visualization(raw_image, samples):
    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    gradcam = samples['gradcams'].reshape(24, 24)

    avg_gradcam = getAttMap(norm_img, gradcam, blur=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(avg_gradcam)
    ax.set_yticks([])
    ax.set_xticks([])
    print('Question: {}'.format(question))


def load_model(model_selection):
    model = AutoModelForCausalLM.from_pretrained(model_selection)
    tokenizer = AutoTokenizer.from_pretrained(model_selection, use_fast=False)
    return model, tokenizer