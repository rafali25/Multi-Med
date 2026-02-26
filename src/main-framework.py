# Generated from: main-framework.ipynb
# Converted at: 2026-02-26T23:11:35.894Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = ""
)

def ask_llama(system_prompt, user_content):
    estimated_input_tokens = (len(system_prompt) + len(user_content)) / 4
    model_limit = 8192
    remaining_tokens = model_limit - estimated_input_tokens - 50
    safe_max_tokens = int(min(512, max(1, remaining_tokens)))
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=safe_max_tokens,
        )
        
    except Exception as e:
        print(str(e))
        return ""
            
    if completion.choices[0].message.content is not None:
        string = completion.choices[0].message.content
        #print(string.strip())
        return string.strip()
    else:
        return ""

def diagnosis(summary, options):
    system =  "You are a medical expert. Your goal is to diagnose the patient."
    user   = f"Patient Vignette:\n\"{summary}\"\n\nYou have to diagnose the patient into one of the following options: [{options}]\n\nRespond in the following format: <thinking></thinking><answer></answer>\nMake sure that the tags should be present.\n\n"
    return ask_llama(system, user)

patients_history = []
summary_history = []
diagnosis_history = []
diagnosis_answers = []
gold_answers = []
gold_options = []

def patient_agent(question, ground_truth_scenario):
    system =  "You are a truthful assistant that understands the patient’s information, and you are trying to answer questions from a medical doctor about the patient. Do not make up facts or tell a lie. Don't think or assume things and just use the context only."
    user   = f'Below is the context paragraph describing the patient and their conditions:\n\"{ground_truth_scenario}\"\n\nQuestion from the doctor: \"{question}\"\nIf no statement answers the question, simply say \"The patient cannot answer this question, please do not ask this question again.\"\nDo not provide any analysis, inference, or implications. Respond with the answer to the question ONLY and NOTHING ELSE.\n\n'
    return ask_llama(system, user)

def summarize(history):
    system = "You are a truthful and accurate medical case summarization assistant. Do NOT make up any facts."
    user   =f"Given below is a chat between a doctor and a patient:\n\"{history}\"\n\nSummarize the patient's information and conditions into a case vignette. Make sure to include as much information as you can while being accurate and without making up facts. You have to respond in the following format: <thinking></thinking><vignette></vignette>\nMake sure that tags should be present\n\n"
    return ask_llama(system, user)

def interrogator_agent(history, options, question_type):
    system =  "You are a medical interviewer. Your goal is to ask the patient clear, one-at-a-time questions."
    user   = f"chat history:\n\"{history}\"\n\nThe patient can have ONLY ONE of the following issues and nothing else: [{options}]\nType of questions to ask next: \"{question_type}\"\n\nGiven the chat history, you have to respond in the following format <thinking></thinking><question></question><justification></justification>\nQuestion is the question to ask the patient and justification is the justification for the question. Do NOT repeat the question already asked. Make sure to ask questions that narrow down the issue to one of the given possible options above. Leave the question tag blank if there is no more questions to ask related to the question type. Make sure that the tags should be present\n\n"
    return ask_llama(system, user)
        
def expert_agent(history, options):
    system =  "You are a medical expert. Your goal is to diagnose the patient."
    user   = f"chat history:\n\"{history}\"\n\nYou have to diagnose the patient into one of the following options: [{options}]\n\nIs the information enough the diagnose the patient? Answer with one of the following: [\"Yes, very confident\", \"Yes, confident\", \"Yes, somewhat confident\", \"No\"]\nIf the information is not enough to diagnose, what kind of questions to ask next?\n\nRespond in the following format: <thinking></thinking><stop_diagnosis></stop_diagnosis><question_type></question_type>\nMake sure that tags should be present.\n\n"
    return ask_llama(system, user)

def driver(facts, options):
    history_justified = ""
    history = ""

    total_questions = 0
    question_type = "Open-Ended Questions Regarding Diagnostics and Demographics Such as Age and Sex"
    
    while total_questions < 50:
        num_questions = 0
        while num_questions < 20:
            iresponse = interrogator_agent(history_justified, options, question_type)
            num_questions += 1
            question = ""
            justification = ""
        
            try:
                ir_start = iresponse.lower().find("<question>")
                ir_end = iresponse.lower().find("</question>")
                if ir_start == -1 or ir_end == -1:
                    break
                    
                question = iresponse[ir_start + 10:ir_end].strip()
            except:
                break

            try:
                j_start = iresponse.lower().find("<justification>")
                j_end = iresponse.lower().find("</justification>")
                if j_end == -1:
                    justification = iresponse[j_start + 15:].strip()
                else:
                    justification = iresponse[j_start + 15:j_end].strip()
            except:
                pass
            
            if question == "":
                break
            else:
                patient_response = patient_agent(question, facts)
                history += f"Doctor: {question}\nPatient: {patient_response}\n\n"
                history_justified += f"Doctor: {question}\njustification: {justification}\n\nPatient: {patient_response}\n\n\n"

        expert_response = expert_agent(history_justified, options)
        to_continue = ""
        question_type = ""

        try:
            tc_start = expert_response.lower().find("<stop_diagnosis>")
            tc_end = expert_response.lower().find("</stop_diagnosis>")
            if tc_start == -1 or tc_end == -1:
                break
                
            to_continue = expert_response[tc_start + 16:tc_end].strip()
        except:
            pass

        if to_continue.strip().lower() in ["yes, very confident", "yes, confident"]:
            break

        try:
            q_start = expert_response.lower().find("<question_type>")
            q_end = expert_response.lower().find("</question_type>")
            if q_start == -1 or q_end == -1:
                break
                
            question_type = expert_response[q_start + 15 : q_end].strip()
        except:
            break
            
        total_questions += num_questions

    patients_history.append([history_justified, history])
    summary = summarize(history)
    summary_history.append(summary)

    s_start = summary.lower().find('<vignette>')
    s_end   = summary.lower().find('</vignette>')
    if s_start == -1:
        return ''

    try:
        if s_end == -1:
            summary = summary[s_start + 10:].strip()
        else:
            summary = summary[s_start + 10:s_end].strip()
    except:
        return ''
    
    diagnosis_response = diagnosis(summary, options)
    diagnosis_history.append(diagnosis_response)
    
    d_start = diagnosis_response.lower().find("<answer>")
    d_end = diagnosis_response.lower().find("</answer>")
    if d_start == -1:
        return ''

    try:
        if d_end == -1:
            diagnosis_answer = diagnosis_response[d_start + 8:].strip()
        else:
            diagnosis_answer = diagnosis_response[d_start + 8: d_end].strip()
    except:
        return ''
        
    return diagnosis_answer

dataset = pd.read_json("/kaggle/input/icraft-md/all_craft_md.jsonl", lines=True)
print(f"Dataframe has {len(dataset)} lines.")

def get_answer(options, answer):
    answer = answer.strip().lower()
    for i in range(len(options)):
        if answer.find(options[i].strip().lower()) != -1:
            return i
            
    return 0

with open('/kaggle/input/icraft-md/patient facts.pickle', 'rb') as file:
    facts = pickle.load(file)

gold = []
generated = []

start = 105
len_dataset = 35

for i in tqdm(range(len_dataset)):
    #print(f'{i}/{len_dataset}')
    row = dataset.iloc[i + start]
    facts_list = facts[i + start]
    facts_str = '\n'.join(facts_list).strip()
    vignette = '. '.join(row['context']).strip()
    options = [row['options']['A'], row['options']['B'], row['options']['C'], row['options']['D']]
    gold_options.append(options)
    options_str = f'"{options[0].strip()}", "{options[1].strip()}", "{options[2].strip()}", "{options[3].strip()}"'
    gold_answers.append(row['answer'])
    answer_index = get_answer(options, row['answer'])
    gen_answer = driver(facts_str, options_str)
    diagnosis_answers.append(gen_answer)
    gen_answer_index = get_answer(options, gen_answer)
    gold.append(answer_index)
    generated.append(gen_answer_index)

accuracy = np.mean(np.array(gold) == np.array(generated))
print(accuracy)

with open('file.pickle', 'wb') as file:
    pickle.dump(patients_history, file)
    pickle.dump(summary_history, file)
    pickle.dump(diagnosis_history, file)
    pickle.dump(diagnosis_answers, file)
    pickle.dump(gold_answers, file)
    pickle.dump(gold_options, file)
    pickle.dump(gold, file)
    pickle.dump(generated, file)