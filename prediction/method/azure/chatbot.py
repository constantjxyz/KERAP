def gen_response(prompt_content, **kwargs):
    model = kwargs.get('model', 'gpt-4o-mini')
    client = kwargs.get('client', None)
    
    try:
        messages = [{"role": "user", "content": prompt_content}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        print(messages)
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        print(f"Error generating response: {e}")
        return None, None, None


class ChatBot:
    def __init__(self, model="gpt-4o-mini", client=None, system_prompt=None):
        self.model = model
        self.client = client
        self.conversation_history = []
        
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})

    def gen_response(self, prompt_content, round=0, use_history=True):
        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt_content})
            messages = self.conversation_history
        else:
            messages = [{"role": "user", "content": prompt_content}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            # 解析模型响应
            assistant_message = response.choices[0].message.content

            if use_history:
                self.conversation_history.append({"role": "assistant", "content": f"Prediction_round_{round}: " + assistant_message})

            return assistant_message, response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception as e:
            print(f"Error generating response: {e}")
            return None, None, None

