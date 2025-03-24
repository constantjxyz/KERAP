from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch

def gen_response(prompt_content, model_name='HuatuoGPT', device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        inputs = tokenizer(prompt_content, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response_text, None, None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None, None, None


class ChatBot:
    def __init__(self, system_prompt=None, model=None, tokenizer=None):
        self.tokenizer = tokenizer
        self.model = model
        self.conversation_history = []
        
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})

    def gen_response(self, prompt_content, round=0, use_history=True):
        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt_content})
            messages = "\n".join([msg["content"] for msg in self.conversation_history])
        else:
            messages = prompt_content

        try:
            prompt_messages = []
            prompt_messages.append({"role": "user", "content": messages})
            
            tokenized_input = self.tokenizer.encode(messages)
            num_tokens = len(tokenized_input)
            response = self.model.HuatuoChat(self.tokenizer, prompt_messages)
            # print(response)
            assistant_message = response
            
            if use_history:
                self.conversation_history.append({"role": "assistant", "content": f"Prediction_round_{round}: " + assistant_message})
            
            return response, num_tokens, len(self.tokenizer.encode(assistant_message))
        except Exception as e:
            print(f"Error generating response: {e}")
            return None, None, None
