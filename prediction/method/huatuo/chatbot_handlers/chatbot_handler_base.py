from method.huggingface.chatbot import ChatBot

class BaseChatBotHandler:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model', 'HuatuoGPT')
        self.bot = ChatBot()

    def prompt_chat(self, ehrdataset, sample_idx, print_conversation=True):
        self.bot = ChatBot(model=self.model, client=self.client, system_prompt=None)
        
        """生成 Prompt、调用 ChatBot，并返回最终预测"""
        examples = self.gen_example_prompt(ehrdataset)
        label_names = ehrdataset.get_label_names()
        
        # 第一轮模型响应
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

    def gen_patient_prompt(self, ehrdataset, sample_idx, examples=[]):
        """生成初始诊断 Prompt"""
        diagnosis_list, _, patient_label_list = ehrdataset.get_sample_data(sample_idx)
        label_list = ehrdataset.get_label_names()
        
        task = ("Your task is to predict whether a patient will develop specific diseases according to the current diagnosis. "
                "Please respond with 'YES' if the patient is likely to develop the specified diseases, or 'NO' if not.\n\n")
        
        diagnosis_text = '; '.join(diagnosis_list)
        diagnosis = f"Diagnosis: {diagnosis_text}\n\n"
        label_text = '; '.join(label_list)
        question = f"Question: Will the patient develop the following {len(label_list)} diseases: {label_text}.\n\n"
        
        output_format = "Please answer in the following format:\n" + "\n".join([f"{label}: <YES or NO>." for label in label_list])

        prompt = task + diagnosis + question + output_format

        # 如果有示例数据，添加示例
        if examples:
            prompt += "\nExamples:\n"
            for i, example in enumerate(examples):
                input_text = example["input"]
                output_text = example["output"]
                prompt += f"Example {i}:\n Diagnosis: {input_text}\n Prediction: {output_text}\n"

        return prompt, patient_label_list

    def combine_example_str(self, train_feature_names, train_labels, label_list):
        """组合训练示例数据"""
        examples = []
        for features, labels in zip(train_feature_names, train_labels):
            diagnosis_text = '; '.join(features)
            input_text = f"Diagnosis: {diagnosis_text}"
            output_text = " ".join([f"{label}: {'YES' if prediction == 1 else 'NO'}." for label, prediction in zip(label_list, labels)])
            examples.append({"input": input_text, "output": output_text.strip()})
        return examples

    def gen_example_prompt(self, ehrdataset):
        """生成用于 Prompt 参考的示例"""
        all_example_feature_names, all_example_labels = [], []
        train_indices = ehrdataset.get_train_indices()

        for example_idx in train_indices:
            patient_feature_names, _, patient_labels = ehrdataset.get_sample_data(example_idx)
            all_example_feature_names.append(patient_feature_names)
            all_example_labels.append(patient_labels)

        return self.combine_example_str(all_example_feature_names, all_example_labels, ehrdataset.get_label_names())

    def ret_prompt(self, response, label_list):
        """解析 ChatBot 生成的预测结果"""
        prediction = []
        if response is None:
            prediction.append(0)
            print("No Response")
        for label in label_list:
            if f"{label}: YES" in response:
                prediction.append(1)
            elif f"{label}: NO" in response:
                prediction.append(0)
            else:
                prediction.append(0)
                print("No Response")
        return prediction
