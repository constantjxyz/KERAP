from method.huatuo.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.huatuo.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, model, tokenizer, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=model, tokenizer=tokenizer)
        label_names = ehrdataset.get_label_names()
        
        patient_prompt, patient_label_list= self.gen_patient_prompt(ehrdataset, sample_idx)
        KG_pos_prompt = ""
        
        response_1, prompt_token_1, completion_token_1  = self.bot.gen_response(KG_pos_prompt + patient_prompt, round=1)
        prediction_1 = self.ret_prompt(response_1, label_names)

        if print_conversation == True:
            print(self.bot.conversation_history)
            print("Labels:", patient_label_list)
            print("Response 1:", response_1)
            print("Prediction:", prediction_1)
            print('\n')
        return [response_1], prediction_1, prediction_1, patient_label_list, prompt_token_1, completion_token_1

    def print_info():
        print(' It directly prompts using EHR information for LLM prediction.')
    
    
    def gen_patient_prompt(self, ehrdataset, sample_idx, examples=[]):
        diagnosis_list, _, patient_label_list = ehrdataset.get_sample_data(sample_idx)
        label_list = ehrdataset.get_label_names()
        
        task = ("Your task is to predict whether a patient will develop specific diseases according to the current diagnosis. "
                "Please respond with 'YES' if the patient is likely to develop the specified diseases, or 'NO' if not.\n\n")
        
        diagnosis_text = '; '.join(diagnosis_list)
        diagnosis = f"Diagnosis: {diagnosis_text}\n\n"
        label_text = '; '.join(label_list)
        question = f"Question: Will the patient develop the following {len(label_list)} diseases: {label_text}.\n\n"
        

        output_format = "请按以下格式输出:\n" + "\n".join([f"<是 或 否>."])
        prompt = task + diagnosis + question + output_format

        if examples:
            prompt += "\nExamples:\n"
            for i, example in enumerate(examples):
                input_text = example["input"]
                output_text = example["output"]
                prompt += f"Example {i}:\n Diagnosis: {input_text}\n Prediction: {output_text}\n"

        return prompt, patient_label_list
    
    
    def ret_prompt(self, response, label_list):
        prediction = []
        if response is None:
            prediction.append(0)
            print("No Response")
        for label in label_list:
            if "是" in response or "YES" in response or "Yes" in response or "很可能" in response or "有可能" in response:
                prediction.append(1)
            elif "否" in response or "没有" in response or "NO" in response or "不" in response or "No" in response:
                prediction.append(0)
            else:
                prediction.append(0)
                print(response)
                print("No Response")
        return prediction