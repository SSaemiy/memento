import os
from dotenv import load_dotenv
import openai
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

load_dotenv()

class VoiceToText:
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.whisper_model = whisper.load_model("small").to(device) 
        
    # 음성 파일 -> 텍스트 파일 변환
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
        
    #텍스트에서 감정 분석
    def analyze_emotion(self, content):

        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, padding=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ 입력도 GPU로
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        filtered_probs = {i: probs[0][i].item() for i in self.selected_labels}
        pred_label = max(filtered_probs, key=filtered_probs.get)

        print(f"[EMOTION] 예측 감정: {self.selected_labels[pred_label]} (score: {filtered_probs[pred_label]:.2f})")
        return self.selected_labels[pred_label] #감정 분석 결과만 리턴

class GptApi:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    #LLM 호출해서 답변 받기
    def RequestAdvice(self, emotion, text):
        reponse = openai.chat.completions.create(
            model = "gpt-4o",
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
            ],
            max_tokens = 50
        )
        return(reponse.choices[0].message.content)

# 실행부
if __name__ == "__main__":
    audio_file = "test_5m.mp3"

    voice_to_text = VoiceToText()
    senti_analysis = SentiAnalysis()
    gpt_advice = GptApi()
    
    content = voice_to_text.transcribe_audio(audio_file)
    emotion = senti_analysis.analyze_emotion(content)
    
    advice = gpt_advice.RequestAdvice(emotion, content)
    print(f"오늘의 조언: {advice}")