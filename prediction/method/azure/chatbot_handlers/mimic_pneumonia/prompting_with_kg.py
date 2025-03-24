from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.azure.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        
        label_names = ehrdataset.get_label_names()
        
        
        KG_pos_prompt = "Do not overestimate the probability of a patient unless certain standards are met. These are key diagnosis factors for pneumonia. Diagnosing pneumonia requires a combination of clinical, radiological, and laboratory findings. Key symptoms include fever, cough (with or without sputum production), shortness of breath, and chest pain, often accompanied by abnormal lung sounds like crackles on auscultation. Imaging, such as chest X-ray or CT scans, confirms lung infiltrates indicative of infection. Laboratory tests, including elevated white blood cell counts, CRP, and procalcitonin, support the diagnosis, while sputum cultures, blood cultures, and PCR tests identify the causative pathogen. Pneumonia can be classified into bacterial, viral, aspiration, eosinophilic, and toxic pneumonitis, among others. Treatment varies based on the etiology, with fluoroquinolones, macrolides, cephalosporins, aminoglycosides, and carbapenems commonly used for bacterial infections, while antiviral agents like acyclovir or zanamivir address viral cases. Supportive therapies such as oxygen therapy, corticosteroids, and immune-boosting agents like vitamin C, zinc, and omega-3 fatty acids can aid recovery. Certain compounds like curcumin, taurine, and theophylline have been found to alleviate symptoms, while natural supplements, including green tea, Schisandra, and Siberian ginseng, may provide additional benefits."
        KG_neg_prompt = "Check your prediction cautiously. Do not overestimate the probability of a patient unless certain provided standards are met. Unrelated factors for diagnosing pneumonia include a vast range of pharmaceuticals, environmental pollutants, industrial chemicals, dietary substances, and biological agents that do not play a direct role in its diagnosis or treatment. These include medications like **antihypertensives, antidepressants, and anticoagulants**, environmental toxins such as **asbestos, ozone, sulfur dioxide, and nitrogen dioxide**, and industrial chemicals like **benzene, formaldehyde, and bisphenol A**. While some substances, like **glucocorticoids, fluoroquinolones, and chemotherapy agents**, may influence immune function or cause lung inflammation, they are not direct diagnostic criteria for pneumonia. Similarly, dietary components like **omega-3 fatty acids, caffeine, and vitamins (A, D, K)**, along with heavy metals such as **cadmium, arsenic, and lead**, do not aid in pneumonia diagnosis. Though these factors can impact respiratory health, proper diagnosis relies on **clinical symptoms (fever, cough, difficulty breathing), imaging (chest X-ray, CT scan), and laboratory tests (sputum culture, PCR, blood tests)** rather than unrelated substances."
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
        print('This chatbot handler is for pneumonia prediction on the MIMIC dataset. It leverages KG information directly on LLM prediction.')
    
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