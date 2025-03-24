from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.azure.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        
        label_names = ehrdataset.get_label_names()
        
        
        KG_pos_prompt = "Chronic kidney disease (CKD) is a progressive condition characterized by declining kidney function, often leading to kidney failure. It shares similarities with conditions such as membranous glomerulonephritis, hypertension, and focal segmental glomerulosclerosis. Key diagnostic indicators include the presence of albuminuria, proteinuria, oliguria, and polyuria, which signal impaired kidney filtration. Systemic manifestations such as edema (including cardiac edema), muscle cramps, fatigue, cachexia, and neurologic symptoms like restless legs syndrome and sleep disorders are also common. Cardiovascular complications, including acute coronary syndrome, angina, heart murmurs, and high cardiac output, further indicate CKD progression. Additional symptoms like pruritus, dyspepsia, anorexia, and taste disorders reflect the systemic impact of the disease. Treatments such as furosemide, calcitriol, and calcium acetate help manage symptoms and slow progression. CKD can advance to end-stage renal disease (ESRD), requiring dialysis or transplantation."

        KG_neg_prompt = "Check your prediction cautiously. Do not overestimate the probability of a patient unless certain provided standards are met. The extracted knowledge graph information highlights several factors that are unrelated to diagnosing chronic kidney disease (CKD). Conditions such as membranous glomerulonephritis, hypertension, restless legs syndrome, focal segmental glomerulosclerosis, anemia, and IgA glomerulonephritis are not classified as CKD, even though some may be associated with kidney health issues. Additionally, CKD does not resemble kidney failure or end-stage renal disease, indicating that while these conditions may be linked, they are distinct entities. Several medications, including cholecalciferol, torasemide, etacrynic acid, cinacalcet, paricalcitol, calcium acetate, deferoxamine, calcitriol, furosemide, and bumetanide, are noted as not treating, alleviating, preventing, or being biomarkers for CKD. Furthermore, these medications do not play a role in CKD pathogenesis or disease progression, emphasizing that their use is not directly relevant to CKD treatment or diagnosis. This information helps refine diagnostic criteria by ruling out unrelated conditions and ineffective treatments, ensuring a more accurate focus on relevant CKD-related factors."
        
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
        print('This chatbot handler is for "chronic kidney disease"  prediction on the MIMIC dataset. It leverages KG information directly on LLM prediction.')
    
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