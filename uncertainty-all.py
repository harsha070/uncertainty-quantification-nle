# Experiments 1, 2, 3
import os
os.environ['OPENAI_API_KEY'] = 'use-your-openai-api-key-here'
import openai
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
tqdm.pandas()


# Model name
# MODEL_NAME = "gpt-3.5-turbo-16k"
# MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "text-davinci-003"
MODEL_NAME = "gpt-4"

CHAT_COMPLETION_API_MODEL_NAMES = ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613",]

CHAT_API_MODEL_NAMES = ["davinci-002", "babbage-002", "text-davinci-003", "text-davinci-002", "text-davinci-001", "text-curie-001", "text-babbage-001", "text-ada-001", "davinci", "curie", "babbage", "ada"]


def get_openai_response(task, question):
    if MODEL_NAME in CHAT_COMPLETION_API_MODEL_NAMES:
        # Call the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user', 
                    'content': (task + '\n' + question).strip(),
                }
            ],
            max_tokens=512,
            n=1,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )
        # Get the response text from the API response
        response_text = response['choices'][0]['message']['content']
        return response_text
    elif MODEL_NAME in CHAT_API_MODEL_NAMES:
        response = openai.Completion.create(
            model=MODEL_NAME,
            prompt=(task + '\n' + question).strip(),
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].text
    else:
        raise Exception
        
        
def get_openai_response_wrapper(args):
    task, question = args
    # try:
    #     return get_openai_response(task=task, question=question)
    # except Exception as e:
    #     print("Call failure once! Retrying")
    #     try:
    #         return get_openai_response(task=task, question=question)
    #     except Exception as e:
    #         print("Call failure twice! Retrying")
    #         return get_openai_response(task=task, question=question)
    # Query (keep trying if we get an error)
    attempts = 0
    try_query = True
    while try_query:
        attempts += 1
        try:
            response = get_openai_response(task=task, question=question)
            try_query = False
        except openai.error.OpenAIError as e:
            print("ERROR! I'm going to sleep for " + str(attempts) + "s")
            print("Error:", str(e))
            time.sleep(attempts + 2)
    return response
        
        

def get_openai_response_batch(task, questions):
    explanations = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        args_list = [(task, question) for question in questions]
        for result in tqdm(executor.map(get_openai_response_wrapper, args_list), total=len(questions)):
            explanations.append(result)
    return explanations


def get_openai_response_temperature(task, question):
    if MODEL_NAME in CHAT_COMPLETION_API_MODEL_NAMES:
        # Call the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user', 
                    'content': (task + '\n' + question).strip(),
                }
            ],
            max_tokens=512,
            n=5,
            temperature=1.0,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )
        # Get the response text from the API response
        response_texts = [choice['message']['content'] for choice in response['choices']]
        return response_texts
    elif MODEL_NAME in CHAT_API_MODEL_NAMES:
        response = openai.Completion.create(
            model=MODEL_NAME,
            prompt=(task + '\n' + question).strip(),
            temperature=1.0,
            max_tokens=512,
            n=5,
        )
        return [choice.text for choice in response.choices]
    else:
        raise Exception


def get_openai_response_temperature_wrapper(args):
    task, question = args
    # try:
    #     return get_openai_response_temperature(task=task, question=question)
    # except:
    #     print("Call failure once! Retrying")
    #     try:
    #         return get_openai_response_temperature(task=task, question=question)
    #     except:
    #         print("Call failure twice! Retrying")
    #         return get_openai_response_temperature(task=task, question=question)
    attempts = 0
    try_query = True
    while try_query:
        attempts += 1
        try:
            response = get_openai_response_temperature(task=task, question=question)
            try_query = False
        except openai.error.OpenAIError as e:
            print("ERROR! I'm going to sleep for " + str(attempts) + "s")
            print("Error:", str(e))
            time.sleep(attempts + 2)
    return response
        

def get_openai_response_temperature_batch(task, questions):
    explanations = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        args_list = [(task, question) for question in questions]
        for result in tqdm(executor.map(get_openai_response_temperature_wrapper, args_list), total=len(questions)):
            explanations.append(result)
    return explanations

dataset_path = str(sys.argv[1])
save_path = dataset_path.replace("input.parquet", f"output-{MODEL_NAME}.parquet")
assert save_path != dataset_path
assert os.path.exists(save_path) is False
data = pd.read_parquet(dataset_path)


# questions = data['experiment_1_question_new'].to_list()
# data['experiment_1_new'] = get_openai_response_batch("", questions)
# print("Experiment 1 new done!")
# data.to_parquet(save_path)

questions = data['experiment_1_question'].to_list()
data['experiment_1'] = get_openai_response_batch("", questions)
print("Experiment 1 done!")
data.to_parquet(save_path)

questions = data['experiment_2_question'].to_list()
data['experiment_2'] = get_openai_response_batch("", questions)
print("Experiment 2 done!")
data.to_parquet(save_path)

questions = data['experiment_3_question'].to_list()
data['experiment_3'] = get_openai_response_batch("", questions)
print("Experiment 3 done!")
data.to_parquet(save_path)

questions = data['experiment_4_question'].to_list()
data['experiment_4'] = [get_openai_response_batch("", paraphrased_questions) for paraphrased_questions in questions]
print("Experiment 4 done!")
data.to_parquet(save_path)

questions = data['experiment_5_question'].to_list()
data['experiment_5'] = [get_openai_response_batch("", paraphrased_questions) for paraphrased_questions in questions]
print("Experiment 5 done!")
data.to_parquet(save_path)

questions = data['experiment_8_question'].to_list()
data['experiment_8'] = get_openai_response_temperature_batch("", questions)
print("Experiment 8 done!")
data.to_parquet(save_path)

questions = data['experiment_9_question'].to_list()
data['experiment_9'] = get_openai_response_temperature_batch("", questions)
print("Experiment 9 done!")

data.to_parquet(save_path)
