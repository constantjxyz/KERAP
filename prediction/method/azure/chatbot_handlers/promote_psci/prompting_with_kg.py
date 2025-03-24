from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
from method.azure.chatbot import ChatBot

class ChatBotHandler(BaseChatBotHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=False):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        
        label_names = ehrdataset.get_label_names()
        
        KG_pos_prompt = "Do not overestimate the probability of a patient unless certain standards are met. These are key diagnosis factors for PSCI: The diagnosis of post-stroke cognitive impairment (PSCI) involves evaluating multiple factors, including structural brain damage, biochemical markers, and environmental influences. Key risk factors identified from the knowledge graph include exposure to neurotoxic substances like lead, arsenic, and pesticides, as well as the presence of oxidative stress markers such as 4-hydroxy-2-nonenal and lipid peroxides. Medications like enalapril and lisinopril have been associated with cognitive dysfunction, while certain compounds, such as docosahexaenoic acid (DHA), folic acid, and Ginkgo biloba, show potential neuroprotective effects. Radiopharmaceutical agents like Florbetaben (18F) and Flutemetamol (18F) are useful in detecting amyloid deposits, which are linked to cognitive decline. The integration of these chemical, pharmacological, and environmental factors with clinical assessment methods, such as neuroimaging and cognitive testing, is essential for accurately diagnosing and managing PSCI."

        KG_neg_prompt = "Check your prediction cautiously. Do not overestimate the probability of a patient unless certain provided standards are met. Unrelated factors in diagnosing post-stroke cognitive impairment (PSCI) include a broad range of environmental toxins, pharmaceuticals, industrial chemicals, and biological compounds that do not contribute to cognitive dysfunction treatment or diagnosis. These factors include heavy metals (e.g., lead, cadmium, mercury), industrial pollutants (e.g., polychlorinated biphenyls, dioxins, benzene), pharmaceuticals (e.g., metformin, haloperidol, tacrine, statins), and dietary compounds (e.g., caffeine, cholesterol, flavonoids). While some of these substances may influence neurotoxicity, oxidative stress, or neurotransmitter function, they do not serve as reliable diagnostic markers or therapeutic agents for PSCI. Additionally, plant extracts, vitamins, and antioxidants like resveratrol, curcumin, and omega-3 derivatives, despite being studied for neuroprotective effects, have shown no direct role in PSCI mitigation. This highlights the need to focus on well-established clinical and pathological factors—such as stroke severity, vascular pathology, neuroinflammation, and neurodegenerative processes—rather than these unrelated environmental and pharmacological exposures for accurate PSCI diagnosis."
        
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
        print('This chatbot handler is for PSCI prediction on the PROMOTE dataset. It leverages KG information for reflection on LLM prediction.')
    
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