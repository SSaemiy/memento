import os
from openai import OpenAI
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class VoiceToText:
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.whisper_model = whisper.load_model("small").to(device) 
        
    def transcribe_audio(self, file_path):
        print(f"[INFO] '{file_path}' 파일을 텍스트로 변환 중...")
        result = self.whisper_model.transcribe(file_path)
        content = result["text"].strip()
        print(f"[TEXT] 인식된 문장: {content}")
        return content
    
class SentiAnalysis:
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model_name = "nlp04/korean_sentiment_analysis_kcelectra"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(device)  # ➤ GPU로 이동
        self.model.eval()
        self.selected_labels = {
            0: '기쁨(행복한)',
            5: '일상적인',
            7: '슬픔(우울한)',
            8: '힘듦(지침)',
            9: '짜증남'
        }
        
    def analyze_emotion(self, content):

        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, padding=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ 입력도 GPU로
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        filtered_probs = {i: probs[0][i].item() for i in self.selected_labels}
        pred_label = max(filtered_probs, key=filtered_probs.get)

        print(f"[EMOTION] 예측 감정: {self.selected_labels[pred_label]} (score: {filtered_probs[pred_label]:.2f})")
        return self.selected_labels[pred_label], filtered_probs[pred_label]

class GptApi:
    def __init__(self):
        self.client = self.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def RequestResponse(self, emotion, text):
        reponse = client.ChatCompletions.create(
            model = "gpt 모델",
            messages= [
                {
                    "role":"system",
                    "content":
                        "당신은 심리상담사입니다."
                },
                {
                    "role":"user",
                    "content":
                        f"{text}를 읽고 {emotion}의 감정에 맞추어 150자 이내로\
                            {text}를 간략하게 2문장 이내로 요약하고 응원하는 말을 작성해주세요."
                }
            ]
        )
        return(reponse.choices[0].message.content)

# 실행부
if __name__ == "__main__":
    audio_file = "test_5m.mp3"

    voice_to_text = VoiceToText()
    senti_Analysis = SentiAnalysis()
    
    content = voice_to_text.transcribe_audio(audio_file)
    senti_Analysis.analyze_emotion(content)