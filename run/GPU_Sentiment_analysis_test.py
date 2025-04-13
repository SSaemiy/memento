import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 🎯 감정 라벨
selected_labels = {
    0: '기쁨(행복한)',
    5: '일상적인',
    7: '슬픔(우울한)',
    8: '힘듦(지침)',
    9: '짜증남'
}

# 🔄 Whisper: 음성 → 텍스트
def transcribe_audio(file_path):
    print(f"[INFO] '{file_path}' 파일을 텍스트로 변환 중...")
    whisper_model = whisper.load_model("small").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ GPU로 이동
    result = whisper_model.transcribe(file_path)
    text = result["text"].strip()
    print(f"[TEXT] 인식된 문장: {text}")
    return text

# 🤖 감정 분석 함수
def analyze_emotion(text):

    model_name = "nlp04/korean_sentiment_analysis_kcelectra"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ GPU로 이동
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ 입력도 GPU로
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    filtered_probs = {i: probs[0][i].item() for i in selected_labels}
    pred_label = max(filtered_probs, key=filtered_probs.get)

    print(f"[EMOTION] 예측 감정: {selected_labels[pred_label]} (score: {filtered_probs[pred_label]:.2f})")
    return selected_labels[pred_label], filtered_probs[pred_label]

# 🚀 실행부
if __name__ == "__main__":
    audio_file = "test_5m.mp3"
    text = transcribe_audio(audio_file)
    analyze_emotion(text)
