# # 챗봇: 고전 방식 ====================================

# import random

# class BasicChatbot:
#     def __init__(self):
#         self.intents = {
#             "greeting": ["안녕", "hi", "hello", "반가워"],
#             "wewather": ["날씨", "weather", "비", "맑아"],
#             "goodbye": ["안녕히", "bye", "goodbye", "잘가"]
#         }
#         self.responses = {
#             "greeting": ["안녕하세요! 무엇을 도와드릴까요?", "반갑습니다!"],
#             "wewather": ["날씨 정보를 조회하겠습니다.", "어느 지역의 날씨를 알고 싶으신가요?"],
#             "goodbye": ["안녕히 가세요!", "좋은 하루 되세요!"],
#             "default": ["죄송합니다. 이해하지 못햇습니다.", "다시 말씀해 주시겠어요?"]
#         }


#     def classify_intent(self, user_input):
#         user_input = user_input.lower()
#         for intent, keywords in self.intents.items():
#             for keyword in keywords:
#                 if keyword in user_input:
#                     return intent
                
#         return "default"
    
    
#     def generate_response(self, intent):
#         responses = self.responses.get(intent, self.responses['default'])
#         return random.choice(responses)
    

#     def chat(self, user_input):
#         intent = self.classify_intent(user_input)
#         response = self.generate_response(intent)
#         return response
    

# bot = BasicChatbot()
# print(bot.chat("안녕하세요"))
# print(bot.chat(""))




# # 챗봇: BERT 기반 의도 분류 ==============================
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# class IntentClassifier:
#     def __init__(self, model_name='klue/bert-base'):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = None
#         self.label_encoder = LabelEncoder()


#     def prepare_data(self, texts, labels):
#         encoded_labels = self.label_encoder.fit_transform(labels)

#         encodings = self.tokenizer(
#             texts,
#             truncation=True,
#             padding=True,
#             max_lenth=128,
#             return_tensors='pt'
#         )

#         print(f"encodings: {encodings}")
#         print(f"encoded_labels: {encoded_labels}")
#         return encodings, encoded_labels
    

#     def train(self, train_texts, train_labels):
#         num_labels = len(set(train_labels))
#         self.model = BertForSequenceClassification.from_pretrained(
#             'klue/bert-base',
#             num_labels = num_labels
#         )

#         train_encodings, train_labels_encoded = self.prepare_data(train_texts, train_labels)

#         class IntentDataset(torch.utils.data.Dataset):
#             def __init__(self, encodings, labels):
#                 self.encodings = encodings
#                 self.labels = labels
            
#             def __getitem__(self, idx):
#                 item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#                 item['labels'] = torch.tensor(self.labels[idx])
#                 return item
            
#             def __len__(self):
#                 return len(self.labels)
            
#         train_dataset = IntentDataset(train_encodings, train_labels_encoded)

#         batch_size = 16
#         epochs = 3
#         learning_rate = 2e-5
#         weight_decay = 0.01

#         train_dataloader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True
#         )

#         optimizer = torch.optim.AdamW(
#             self.model.parameters(),
#             lr=learning_rate,
#             weight_decay=weight_decay
#         )

#         self.model.train()

#         for epoch in range(epochs):
#             print(f"Epoch {epoch + 1}/{epochs}")
#             total_loss = 0

#             for batch in train_dataloader:
#                 optimizer.zero_grad()

#                 outputs = self.model(**batch)
#                 loss = outputs.loss

#                 loss.backward()

#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

#                 optimizer.step()

#                 total_loss += loss.item()

#             avg_loss = total_loss / len(train_dataloader)
#             print(f"Average Loss: {avg_loss:.4f}")

#         print("훈련 완료")


#     def predict(self, text):
#         inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         print(f"inputs: {inputs}")

#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

#         predicted_class = torch.argmax(predictions, dim=1).item()

#         confidence = torch.max(predictions).item()

#         intent = self.label_encoder.inverse_transform([predicted_class])[0]

#         return intent, confidence
    
# train_texts = [
#     "안녕하세요", "반갑습니다", "hello",
#     "날씨가 어때요?", "비가 와요?", "맑나요?",
#     "주문하고 싶어요", "메뉴 보여줘", "배달 가능해?"
# ]

# train_labels = [
#     "greeting", "greeting", "greeting",
#     "weather", "weather", "weather",
#     "order", "order", "order"
# ]



# classifier = IntentClassifier()
# classifier.train(train_texts, train_labels)
# intent, confidence = classifier.predict("오늘 날씨 어때요?")
# print(f"의도: {intent}, 신뢰도: {confidence:.2f}")








# # OPENAI: 함수 호출 챗봇 =======================================

# import os
# import json
# from datetime import datetime
# from openai import OpenAI
# import dotenv
# dotenv.load_dotenv()
# from OpenAIChatbot import OpenAIChatbot


# class AdvancedChatbot(OpenAIChatbot):
#     def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo'):
#         super().__init__(api_key, model)
#         self.available_functions = {
#             "get_weather": self.get_weather,
#             "calculate": self.calculate,
#             "get_time": self.get_time
#         }


#     def get_weather(self, location: str) -> str:
#         weather_data = {
#             "서울": "맑음, 15도",
#             "부산": "흐림, 18도",
#             "대구": "비, 12도"
#         }
#         return weather_data.get(location, "해당 지역의 날씨 정보를 찾을 수 없습니다.")
    

#     def calculate(self, expression: str) -> str:
#         try:
#             result = eval(expression)
#             return f"{expression} = {result}"
#         except:
#             return "계산할 수 없는 식입니다."
        

#     def get_time(self) -> str:
#         return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    

#     def get_function_response(self, user_message: str) -> str:
#         tools = [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "get_weather",
#                     "description": "특정 지역의 날씨 정보를 조회합니다",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "location": {
#                                 "type": "string",
#                                 "description": "날씨를 조회할 지역명"
#                             }
#                         },
#                         "required": ["location"]
#                     }
#                 }
#             },

#             {
#                 "type": "function",
#                 "function": {
#                     "name": "calculate",
#                     "description": "수학 계산을 수행합니다",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "expression": {
#                                 "type": "string",
#                                 "description": "게산할 수식"
#                             }
#                         },
#                         "required": ["expression"]
#                     }
#                 }
#             },

#             {
#                 "type": "function",
#                 "function": {
#                     "name": "get_time",
#                     "description": "현재 시간을 조회합니다",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {}
#                     }
#                 }
#             }
#         ]

#         self.add_message("user", user_message)
#         messages = [self.system_message] + self.conversation_history

#         try:
#             response = self.client.chat.completions.create(
#                 model = self.model,
#                 messages = messages,
#                 tools = tools,
#                 tool_choice = "auto"
#             )

#             if message.tool_calls:
#                 tool_call = message.tool_calls[0]
#                 function_name = tool_call.function.name
#                 arguments = json.loads(tool_call.function.arguments)

#                 if function_name in self.available_functions:
#                     function_response = self.available_functions[function_name](**arguments)
#                     self.add_message("assistant", "")
#                     self.conversation_history[-1]["tool_calls"] = message.tool_calls

#                     self.conversation_history.append({
#                         "role": "tool",
#                         "content": function_response,
#                         "tool_call_id": tool_call.id
#                     })

#                     final_response = self.client.chat.completions.create(
#                         model = self.model,
#                         messages = [self.system_message] + self.conversation_history
#                     )

#                     final_message = final_response.choices[0].message.content
#                     self.add_message("assistant", final_message)

#                     return final_message
                
#                 else:
#                     assistant_message = message.content
#                     self.add_message("assistant", assistant_message)
#                     return assistant_message
                
#         except Exception as e:
#             return f"오류가 발생했습니다: {str(e)}"
        


# api_key = os.environ.get("OPENAI_API_KEY")
# advanced_bot = AdvancedChatbot(api_key)
# advanced_bot.set_system_prompt("당신은 친절한 비서입니다. 함수 호출을 통해 날씨 정보, 계산 결과, 현재 시간을 조회할 수 있습니다.")

# # 날씨 조회 테스트
# response = advanced_bot.get_function_response("서울 날씨 어때?")
# print(f"봇: {response}")

# # 계산 테스트
# response = advanced_bot.get_function_response("25 곱하기 4는 얼마야?")
# print(f"봇: {response}")

# # 날씨 조회 테스트
# response = advanced_bot.get_function_response("지금 몇 시야?")
# print(f"봇: {response}")




# OPENAI: 대화 상태 관리 =====================================
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import uuid
from OpenAIChatbot import OpenAIChatbot
import os
import dotenv
from EntityExtractor import EntityExtractor


dotenv.load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class DialogState:

    session_id: str
    entities: Dict[str, Any] = field(default_factory=dict)
    context_stack: List[Dict] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    
    def add_turn(self, user_input: str, bot_response: str, entities: Dict = None):

        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "entities": entities or {}
        }
        self.conversation_history.append(turn)

        if entities:
            self.entities.update(entities)

        self.last_updated = datetime.now()


    def get_context(self, turns: int = 3) -> List[Dict]:
        return self.conversation_history[-turns:] if self.conversation_history else []
    

    def clear_context(self):
        self.conversation_history = []
        self.entities = {}
        self.context_stack = []



class ConversationManager:

    def __init__(self):
        self.sessions: Dict[str, DialogState] = {}
        self.entity_extractor = EntityExtractor()


    def create_session(self, user_id: str = None) -> str:
        session_id = user_id or str(uuid.uuid4())
        self.sessions[session_id] = DialogState(session_id=session_id)

        return session_id
    

    def get_session(self, session_id: str) -> Optional[DialogState]:
        return self.sessions.get(session_id)
    

    def process_message(self, session_id: str, user_input: str, chatbot) -> str:

        if session_id not in self.sessions:
            self.create_session(session_id)

        dialog_state = self.sessions[session_id]
        entities = self.entity_extractor.extract_entities(user_input)
        entity_dict = {ent['label']: ent['text'] for ent in entities}
        context_prompt = self._build_context_prompt(dialog_state, user_input)
        response = chatbot.get_response(context_prompt)

        dialog_state.add_turn(
            user_input=user_input,
            bot_response=response,
            entities=entity_dict
        )

        return response
    

    def _build_context_prompt(self, dialog_state: DialogState, current_input: str) -> str:

        context_info = []

        recent_context = dialog_state.get_context(turns=3)
        if recent_context:
            context_info.append("이전 대화:")
            for turn in recent_context:
                context_info.append(f"사용자: {turn['user_input']}")
                context_info.append(f"봇: {turn['bot_response']}")

            if dialog_state.entities:
                context_info.append(f"기억된 정보: {dialog_state.entities}")

            if dialog_state.user_profile:
                context_info.append(f"사용자 정ㅈ보: {dialog_state.user_profile}")

            context_info.append(f"현재 질문: {current_input}")

            return "\n".join(context_info)
        

    def update_user_profile(self, session_id: str, profile_data: Dict):

        if session_id in self.sessions:
            self.sessions[session_id].user_profile.update(profile_data)
    

    def get_conversation_summary(self, session_id: str) -> str:

        dialog_state = self.sessions.get(session_id)
        if not dialog_state or not dialog_state.conversation_history:
            return "대화 내역이 없습니다."
        
        summary_parts = []
        summary_parts.append(f"세션 ID: {session_id}")
        summary_parts.append(f"대화 턴 수: {len(dialog_state.conversation_history)}")
        summary_parts.append(f"추출된 개체: {dialog_state.entities}")

        return "\n".join(summary_parts)
    



conv_manager = ConversationManager()

chatbot = OpenAIChatbot(api_key)
chatbot.set_system_prompt("당신은 친절한 비서입니다.")


session_id = conv_manager.create_session("user123")

response1 = conv_manager.process_message(session_id, "안녕하세요", chatbot)
print(f"봇1: {response1}")

response2 = conv_manager.process_message(session_id, "제 이름은 김철수입니다", chatbot)
print(f"봇2: {response2}")


conv_manager.update_user_profile(session_id, {"name": "김철수", "age": 30})

response3 = conv_manager.process_message(session_id, "아까 제가 뭐라고 했죠?", chatbot)
print(f"봇3: {response3}")

response4 = conv_manager.process_message(session_id, "제 나이에 맞는 취미를 추천해주세요", chatbot)
print(f"봇4: {response4}")


# 대화 요약
summary = conv_manager.get_conversation_summary(session_id)
print(f"대화 요약: \ㅜ{summary}")