class memento_whisper:
    def __init__(self):
        whisper_model = whisper.load_model("small").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # ➤ GPU로 이동