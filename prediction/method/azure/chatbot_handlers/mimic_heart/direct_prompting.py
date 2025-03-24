from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.azure.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        
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
        print('This chatbot handler is for "Congestive heart failure; nonhypertensive" prediction on the MIMIC dataset. It directly prompts using EHR information on LLM prediction.')