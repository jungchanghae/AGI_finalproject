from lavis.models import load_model_and_preprocess
import time
from datasets import load_dataset
import torch
from utils.vqa_model import load_model, postprocess_Answer
from utils.preprocess import get_class_label, create_template, evaluate_answer, load_class_label, label_int2str
from tqdm import tqdm
import random 

if __name__ == "__main__":
    random.seed(113)
    device = torch.device("cuda:0")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base",
                                                                      is_eval=True, device=device)

    print("Loading Large Language Model (LLM)...")
    llm_model, tokenizer = load_model('EleutherAI/gpt-neo-1.3B')
    # llm_model, tokenizer = load_model('facebook/opt-6.7b')  # ~13G (FP16)
    llm_model = llm_model.to("cuda:0")

    # llm_model, tokenizer = load_model('facebook/opt-6.7b')  # ~13G (FP16)
    # llm_model, tokenizer = load_model('facebook/opt-13b') # ~26G (FP16)
    # llm_model, tokenizer = load_model('facebook/opt-30b') # ~60G (FP16)
    # llm_model, tokenizer = load_model('facebook/opt-66b') # ~132G (FP16)

    print("Loading Dataset")
    df = load_dataset("food101", split="validation")
    print(df)
    df = df.shuffle(seed=113)
    df = df.shard(num_shards=20, index=0)
    # df = load_dataset("taesiri/imagenet-hard")['validation']
    # class_labels = get_class_label(df)
    class_labels = load_class_label('/workspace/project/AGI_finalproject/dataset/food101/dataset_infos.json')
    print(len(class_labels))
    # ask a random question.

    outputs_list = []  # 결과 모음
    matched_num, sub_matched_result = 0, 0
    print("Inference Start")
    for row in tqdm(df):
        # gold_label = row['english_label'][0]
        gold_label = label_int2str(class_labels,row['label'])
        question, true_index = create_template(class_labels, gold_label,candidate_num=4)

        if row['image'].mode != "RGB":
            image = row['image'].convert("RGB")
        else:
            image = row['image']

        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": [question]}
        # IMG 2 Question Matching
        samples = model.forward_itm(samples=samples)
        # Image Captioning
        samples = model.forward_cap(samples=samples, num_captions=50, num_patches=20)
        # Question Generation
        samples = model.forward_qa_generation(samples)
        # Prepare prompts for LLM
        Img2Prompt = model.prompts_construction(samples)
        # Img2Prompt_input = tokenizer(Img2Prompt, padding='longest', truncation=True, return_tensors="pt").to(device)
        Img2Prompt_input = tokenizer(Img2Prompt, truncation=True, return_tensors="pt").to(device)
        # print(len(Img2Prompt_input['input_ids'][0]))
        assert (len(Img2Prompt_input.input_ids[0]) + 20) <= 2048

        # Generate Answer Referring Prompts
        Img2Prompt_input = Img2Prompt_input.to("cuda:0")
        outputs = llm_model.generate(input_ids=Img2Prompt_input.input_ids,
                                     attention_mask=Img2Prompt_input.attention_mask,
                                     max_length=20 + len(Img2Prompt_input.input_ids[0]),
                                     return_dict_in_generate=True,
                                     output_scores=True
                                     )

        generated_answer = tokenizer.batch_decode(outputs.sequences[:, len(Img2Prompt_input.input_ids[0]):])
        pred_label = postprocess_Answer(generated_answer)
        pred_answer = f"{Img2Prompt} {postprocess_Answer(generated_answer)}"
        outputs_list.append(pred_answer)

        # 정답 비율
        matched_num, sub_matched_result = evaluate_answer(pred_label, gold_label, matched_num, sub_matched_result)

    # 결과 저장
    with open(f"./results/results_{time.time_ns()}.txt", "w", encoding="utf-8") as f:
        f.write(f"Total: {len(df)}, Exact Accuracy: {matched_num/len(df)}, "
                f"Sub_matched Accuracy: {(matched_num + sub_matched_result)/len(df)}"+"\n\n")
        for row in outputs_list:
            f.write(row + "\n\n")
