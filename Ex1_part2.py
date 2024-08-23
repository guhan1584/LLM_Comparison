import os
import pandas as pd
from gpt4all import GPT4All
import time
import requests
import redis
from dotenv import load_dotenv

load_dotenv()
wolfram_key = os.environ.get('api_key_wolfram')

#System templates for models
system_template_orcamini  = '''A chat between a curious User and an artificial intelligence assistant. give a straight answer, WITHOUT any addition text.
                            DO NOT anything! JUST the answer to the question, ONLY ANSWER.
                            DO NOT write "AI" or any other things, DO NOT generate questions or prompting as a "User", JUST write the answer to the question. SIMPLE.'''

system_template_falcon = '''A chat between a curious user and an artificial intelligence assistant. give a straight answer, WITHOUT any addition text.
                            DO NOT anything! JUST the answer to the question, ONLY ANSWER.
                            DO NOT write "Assitant" or any other things, DO NOT generate questions or prompting as a "Human", JUST write the answer to the question. SIMPLE.'''

system_template_minstral_instruct = '''You are an expert analyst assistant. Your task is to evaluate similarity between two answers and generate a float value between 0 and 1.0. 
                        For examlpe:
                        #User: question - what is the capital of Israel?
                        answer 1 - Jerusalem , answer 2 - Tel Aviv
                        #Assistant: 0.7
                        '''

#Prompt template for the judge model
prompt_template_minstral_instruct = '''[INS] {0} [/INS]'''

def generate_response(request , model_name , model_id): # generate each responds to a question based on the question and the model
    prompt = f'###User: Just a concise answer for the quesion - {request}'
    start_time = time.time()
    output = model_id.generate(prompt, max_tokens=20,).strip("\n")
    end_time = time.time()
    model_response_time = int((end_time - start_time)*1000)
    return (request , output , model_response_time, model_name)

def judging (question_df , LLM_answer , wolfram_knowledge): # the function that calculate the correctness of the LLM
    prompt1 = f'''here is Here is a question - {question_df}. DO NOT try to solve the question.
    Here are two answers to that question:
    1. {wolfram_knowledge} and 2. {LLM_answer}.
    Evaluate their similarity between the two answers on a scale of 0 to 1.0 . Return as an output - ONLY 
    the evalutaion score (float value). DO NOT give any explantions or addittion texts, ONLY the float value.
    The similarity value is: '''
    with Mistral_instruct.chat_session(system_template_minstral_instruct, prompt_template_minstral_instruct):
        judge_response = Mistral_instruct.generate( prompt1 ,max_tokens=5, temp=0).strip("\n")
    return judge_response

def checking_redis_db (request, wolfram_key, ): # a fucntion that checks if the answer is in redis
    wolfram_client_temp = f'https://api.wolframalpha.com/v1/result?appid={wolfram_key}&i={request}'
    wolfram_answer_temp = requests.get(wolfram_client_temp).text.strip("\n")
    redis_client.set(request, wolfram_answer_temp)
    redis_client.expire(request, 60*60*4)

file_path = r"C:\Users\guhan\Desktop\assignments\AI_developing\Assignments_environment\venv\Assignment_1\General_Knowledge_Questions.csv"
df = pd.read_csv(file_path)

question_with_response_wolfram = []  # keep tracking on the amount of question the wolfram alpha has an answer on
question_and_response_falcon = []
question_and_response_orcamini = []
total_score_orcamini = 0
total_score_falcon = 0
redis_client = redis.Redis()
#redis_client.flushdb()

for i in range(len(df)):
    request = (df['Question'][i]).strip("\n")
    if((redis_client.get(request) is None) and (redis_client.get(request) != "not in use")): # To understand if we need to check what is the answer or we know already that wolfram doesn't know the answer
        wolfram_client = f'https://api.wolframalpha.com/v1/result?appid={wolfram_key}&i={request}'
        wolfram_answer = requests.get(wolfram_client).text.strip("\n")
        print(i)
        if((wolfram_answer != 'Wolfram|Alpha did not understand your input') and (wolfram_answer != 'No short answer available')):
            redis_client.set(request, wolfram_answer)
            redis_client.expire(request, 60*60*4)
            question_with_response_wolfram.append(request) # An array that keep track on the questions that have an answer on wolfram alpha, in order to optimize the number of iteration per LLM model
            print(f"question: {request} , wolfram_answer: {wolfram_answer}")
        else:
            redis_client.set(request, "not in use") # I put the value "not in use" in redis for each question that wolfram alpha doesnt know how to answer on
            redis_client.expire(request, 60*60*4)
    else:
        #print(redis_client.get(request))
        if(redis_client.get(request).decode('utf-8') != "not in use" and (redis_client.get(request) is not None)): # verifying that we will know how much question has answer in order to save time and unnecessary iterations
            question_with_response_wolfram.append(request)
#if(len(question_with_response_wolfram) == 50): # It was a test to check if the conditions above fulfill their purpose
  #  exit()
#print(f"number of questions with wolfram answers: {len(question_with_response_wolfram)}") # Checking if indeed we have 23 questions wit response

#Model Initial
Gpt_falcon = GPT4All("gpt4all-falcon-q4_0.gguf")
Mistral_instruct = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")

#with Gpt_falcon.chat_session(system_template_falcon, prompt_template_mistral_openorca_and_falcon ):       
for i in range(len(question_with_response_wolfram)):
    FalconOutput = generate_response(question_with_response_wolfram[i] , "Falcon" , Gpt_falcon)
    FalconOutput_0_temp = FalconOutput[0].strip("\n")
    FalconOutput_1_temp  = FalconOutput[1].strip("\n")
    if(redis_client.get(question_with_response_wolfram[i]) is None):
        checking_redis_db(question_with_response_wolfram[i], wolfram_key)
    Falcon_answer = {
        'DfQuestion' : FalconOutput_0_temp,
        'ModelName' : FalconOutput[3],
        'ModelAnswer' : FalconOutput_1_temp,
        'ResponseTime' : FalconOutput[2],
        'Correctness' : float(judging(FalconOutput_0_temp , FalconOutput_1_temp , redis_client.get(question_with_response_wolfram[i]).decode('utf-8')))
    }
    #print(f'{Falcon_answer} , {i}') # checking the model resopond
    total_score_falcon += float(Falcon_answer['Correctness'])
    question_and_response_falcon.append(Falcon_answer)
    #if(i == 5): # for video and debugging purposes
      #  break

#Model Initial
Orca_mini = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

#with Orca_mini.chat_session(system_template_orcamini, prompt_template_orca_mini):       
for i in range(len(question_with_response_wolfram)):
    OrcaMiniOutput = generate_response(question_with_response_wolfram[i], "OrcaMini" , Orca_mini)
    OrcaMiniOutput_0_temp  = OrcaMiniOutput[0].strip("\n")
    OrcaMiniOutput_1_temp   = OrcaMiniOutput[1].strip("\n")
    if(redis_client.get(question_with_response_wolfram[i]) is None):
        checking_redis_db(question_with_response_wolfram[i], wolfram_key)
    OracMini_answer = {
        'DfQuestion' : OrcaMiniOutput_0_temp,
        'ModelName' : OrcaMiniOutput[3],
        'ModelAnswer' : OrcaMiniOutput_1_temp,
        'ResponseTime' : OrcaMiniOutput[2],
        'Correctness' : float(judging(OrcaMiniOutput_0_temp , OrcaMiniOutput_1_temp , redis_client.get(question_with_response_wolfram[i]).decode('utf-8')))
    }
    #print(f'{OracMini_answer} , {i}') # checking the model resopond
    total_score_orcamini += OracMini_answer['Correctness']
    question_and_response_orcamini.append(OracMini_answer)
   # if(i == 5): # for video and debugging purposes
    #   break

lowest_rated_question_and_answer_falcon = min(question_and_response_falcon , key=lambda x: x['Correctness'])
lowest_rated_question_and_answer_orcamini = min(question_and_response_orcamini , key=lambda x: x['Correctness'])
avg_score_falcon = total_score_falcon/len(question_and_response_falcon)
avg_score_orcamini = total_score_orcamini/len(question_and_response_orcamini)
print(f'The number of questions that wolfram alpha had answer is: {len(question_with_response_wolfram)}')
print(f'The average answer rating of falcon is: {avg_score_falcon}')
print(f'The average answer rating of orca-mini is: {avg_score_orcamini}')
print(f'The lowest rated question and answer of falcon is: {lowest_rated_question_and_answer_falcon['DfQuestion']} , {lowest_rated_question_and_answer_falcon['ModelAnswer']}')
print(f'The The lowest rated question and answer of orca-mini is: {lowest_rated_question_and_answer_orcamini["DfQuestion"]} , {lowest_rated_question_and_answer_orcamini["ModelAnswer"]}')
