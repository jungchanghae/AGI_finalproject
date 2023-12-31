from datasets import load_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import random
from io import BytesIO
from openai import OpenAI

client = OpenAI()
def get_LLM_prompt(img_string):

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Create prompt of image to feed to VQA"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_string}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content


def convert_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

# Load the dataset
dataset = load_dataset("food101", split="validation")
dataset = dataset.shuffle(seed=113)
dataset = dataset.shard(num_shards=20, index=0)
# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

correct_answers = 0
total_questions = 0

for i, data in enumerate(dataset):
    if i == 300:
        break
    correct_label = dataset[i]
    print(correct_label)

    # Assuming you have a way to get the list of class names
    class_names = dataset.features['label'].names

    # Get an example from the dataset

    # Get the label index
    label_index = data['label']

    # Retrieve the label name
    label_name = class_names[label_index]

    # Create answer options including the correct answer
    random_choices = random.sample([name for name in class_names if name != label_name], 3)
    answer_options = random_choices + [label_name]
    random.shuffle(answer_options)
    correct_answer_index = answer_options.index(label_name) + 1  # Adding 1 because options are 1-indexed

    # Formulate the question
    question = f'Please help to answer what is this picture. Choose your number between 2,3,4,1? 1: "{answer_options[0]}", 2: "{answer_options[1]}", 3: "{answer_options[2]}", 4: "{answer_options[3]}" Thank you'

    # Process the image and question
    image = convert_to_rgb(data['image'])
    buffered = BytesIO()
    b_image = image
    b_image.save(buffered, format="JPEG")  # Or PNG, depending on your image format
    # Encode the bytes buffer to Base64
    inputs = processor(image, question, return_tensors="pt")

    # Get model prediction
    out = model.generate(**inputs)
    try:
        model_answer = int(processor.decode(out[0], skip_special_tokens=True)[0])
        # model_answer = processor.decode(out[0], skip_special_tokens=True)
        print("Model Answer: ", model_answer)
        print("Real option: ", answer_options[correct_answer_index-1])
        print(processor.decode(out[0], skip_special_tokens=True))
        # Evaluate the model's answer
        if model_answer == correct_answer_index:
            correct_answers += 1
    except ValueError:
        print(f"Model output not an integer for image {i}: {processor.decode(out[0], skip_special_tokens=True)[0]}")
        # continue


    # Evaluate the model's answer
    total_questions += 1

# Calculate accuracy
accuracy = correct_answers / total_questions
print(f"Model Accuracy: {accuracy * 100:.5f}%")
