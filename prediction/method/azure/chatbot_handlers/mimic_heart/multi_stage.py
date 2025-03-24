from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.azure.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        label_names = ehrdataset.get_label_names()
        
        
        KG_pos_prompt = ""
        
        KG_prompt = KG_pos_prompt
        patient_prompt, patient_label_list= self.gen_patient_prompt(ehrdataset, sample_idx, KG_prompt)
        response_1, prompt_token_1, completion_token_1  = self.bot.gen_response(patient_prompt, round=1)
        prediction_1 = self.ret_prompt(response_1, label_names)
        
        KG_neg_prompt = "Check your prediction cautiously. Do not overestimate the probability of a patient unless certain provided standards are met. Please answer in the following format: Congestive heart failure; nonhypertensive: <YES or NO>."
        
        response_2, prompt_token_2, completion_token_2  = self.bot.gen_response(KG_neg_prompt, round=2)
        prediction_2 = self.ret_prompt(response_2, label_names)

        if print_conversation == True:
            print(self.bot.conversation_history)
            print("Labels:", patient_label_list)
            print("Response 1:", response_1)
            print("Response 2:", response_2)
            print("Prediction:", prediction_2)
            print('\n')

        return [response_1, response_2], prediction_1 + prediction_2, prediction_2, patient_label_list, prompt_token_1 + prompt_token_2, completion_token_1 + completion_token_2
    
    def print_info():
        print('This chatbot handler is for "Congestive heart failure; nonhypertensive" prediction on the MIMIC dataset. It leverages multi-stage prompts on LLM prediction.')
    
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