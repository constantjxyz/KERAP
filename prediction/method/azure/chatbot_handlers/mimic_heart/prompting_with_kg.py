from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.azure.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        label_names = ehrdataset.get_label_names()
        
        
        KG_pos_prompt = "Diagnosing **Congestive Heart Failure (CHF) without hypertension** involves assessing clinical symptoms, medical history, and potential contributing factors. Key indicators include **dyspnea (shortness of breath), fatigue, peripheral edema, pulmonary congestion**, and reduced exercise tolerance. **Echocardiography** is crucial for evaluating cardiac function, particularly **ejection fraction (EF)** to differentiate between heart failure with preserved (HFpEF) or reduced ejection fraction (HFrEF). Laboratory tests such as **brain natriuretic peptide (BNP) or N-terminal pro-BNP (NT-proBNP)** help confirm heart failure. Additionally, knowledge graph insights reveal that various **medications and substances (e.g., NSAIDs, corticosteroids, beta-blockers, chemotherapeutic agents, and immunosuppressants)** may induce or exacerbate CHF. Other contributing factors include **thyroid dysfunction (e.g., levothyroxine, liothyronine), electrolyte imbalances (sodium, iron), and metabolic disorders (glucose abnormalities)**. Diagnosis is finalized through **clinical evaluation, imaging (e.g., chest X-ray, MRI, or cardiac catheterization), and exclusion of other causes** such as primary hypertensive heart disease or valvular disorders."

        
        KG_neg_prompt = "Check your prediction cautiously. Do not overestimate the probability of a patient unless certain provided standards are met. Diagnosing **congestive heart failure (CHF) without hypertension** involves considering unrelated factors that may contribute to or mimic the condition. These include **pulmonary diseases** (such as chronic obstructive pulmonary disease [COPD] or pulmonary embolism), **renal dysfunction** (which can cause fluid retention and exacerbate heart failure symptoms), **hepatic disorders** (like cirrhosis, leading to ascites and peripheral edema), and **endocrine abnormalities** (such as thyroid dysfunction, which can affect cardiac output). Additionally, **anemia** and **nutritional deficiencies** may cause fatigue and dyspnea, resembling CHF. Psychological conditions like **anxiety** can also mimic symptoms of heart failure, including palpitations and shortness of breath. Diagnostic differentiation requires clinical assessment, imaging, and laboratory tests to rule out these unrelated but potentially confounding factors."
        
        KG_prompt = KG_pos_prompt + KG_neg_prompt
        patient_prompt, patient_label_list= self.gen_patient_prompt(ehrdataset, sample_idx, KG_prompt)
        response_1, prompt_token_1, completion_token_1  = self.bot.gen_response(patient_prompt, round=1)
        prediction_1 = self.ret_prompt(response_1, label_names)

        if print_conversation == True:
            print(self.bot.conversation_history)
            print("Labels:", patient_label_list)
            print("Response:", response_1)
            print("Prediction:", prediction_1)
            print('\n')

        return [response_1], prediction_1, prediction_1, patient_label_list, prompt_token_1, completion_token_1

    def print_info():
        print('This chatbot handler is for "Congestive Heart Failure (CHF) without hypertension"  prediction on the MIMIC dataset. It leverages KG information directly on LLM prediction.')
    
    def gen_patient_prompt(self, ehrdataset, sample_idx, KG_prompt, examples=[], ):
        diagnosis_list, _, patient_label_list = ehrdataset.get_sample_data(sample_idx)
        label_list = ehrdataset.get_label_names()
        
        task = ("Your task is to predict whether a patient will develop specific diseases according to the current diagnosis. "
                "Please respond with 'YES' if the patient is likely to develop the specified diseases, or 'NO' if not.\n\n")
        
        diagnosis_text = '; '.join(diagnosis_list)
        diagnosis = f"Diagnosis: {diagnosis_text}\n\n"
        label_text = '; '.join(label_list)
        question = f"Question: Will the patient develop the following {len(label_list)} diseases: {label_text}.\n\n"
        
        
        output_format = "Please answer in the following format:\n" + "\n".join([f"{label}: <YES or NO>." for label in label_list])

        prompt = task + diagnosis + question + KG_prompt + output_format

        if examples:
            prompt += "\nExamples:\n"
            for i, example in enumerate(examples):
                input_text = example["input"]
                output_text = example["output"]
                prompt += f"Example {i}:\n Diagnosis: {input_text}\n Prediction: {output_text}\n"

        return prompt, patient_label_list