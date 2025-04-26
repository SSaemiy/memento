class senti_analysis:
    def __init__(self):
        model_name = "nlp04/korean_sentiment_analysis_kcelectra"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ GPU로 이동
        
    def analyze_emotion(content):
        model.eval()

        inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ 입력도 GPU로
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        filtered_probs = {i: probs[0][i].item() for i in selected_labels}
        pred_label = max(filtered_probs, key=filtered_probs.get)

        print(f"[EMOTION] 예측 감정: {selected_labels[pred_label]} (score: {filtered_probs[pred_label]:.2f})")
        return selected_labels[pred_label], filtered_probs[pred_label]