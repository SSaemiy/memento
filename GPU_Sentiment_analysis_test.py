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
            0: '기쁨',
            5: '중립',
            7: '슬픔',
            8: '힘듦',
            9: '화남'
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
                        "당신은 친절한 심리상담전문가입니다. 사용자의 감정에 공감해야합니다."
                },
                {
                    "role":"user",
                    "content":
                        f"{text}가 비어있거나 문장을 요약할 수 없을 경우 바빠서 일기를 쓰지 못했던 상황을 가정하고 응원하는 말을 작성해주세요.\
                        {text}가 한 문장일 경우 요약하지 않고 {text}와 {emotion}에 맞추어 응원하고 공감하는 말을 150자 이내로 작성합니다.\
                        {text}에 요약할 문장이 있는 경우 {text}를 읽고 {text}를 간략하게 1문장 이내로 요약해주세요.\
                        요약한 문장은 ~하셨군요 로 끝맺어야합니다.\
                        {emotion}의 감정에 맞추어 사용자를 응원하는 말을 150자 이내로 작성해주세요.\
                        답변에는 줄넘김이 없어야합니다."
                }
            ],
            max_tokens = 150
        )
        return(reponse.choices[0].message.content)

# 실행부
if __name__ == "__main__":
    #audio_file = "test_5m.mp3"

    #voice_to_text = VoiceToText()
    #senti_analysis = SentiAnalysis()
    gpt_advice = GptApi()
    
    content = "화난다."
    #content = voice_to_text.transcribe_audio(audio_file)

    emotion = "화남"
    #emotion = senti_analysis.analyze_emotion(content)
    
    print(f"일기 내용 : {content}")

    advice = gpt_advice.RequestAdvice(emotion, content)
    print(f"오늘의 조언: {advice}")