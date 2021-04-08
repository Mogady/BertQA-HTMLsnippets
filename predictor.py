from model import predict
from process import post_process, extract_text
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from starlette.responses import JSONResponse


class PythonPredictor:
    def __init__(self, config):
        device = 0 if torch.cuda.is_available() else -1
        print(f"using device: {'cuda' if device == 0 else 'cpu'}")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    def predict(self, payload):
        question = payload['question'].replace("\\/", "/").encode().decode('unicode_escape')
        html_article = payload['article'].replace("\\/", "/").encode().decode('unicode_escape')
        context = extract_text(html_article)

        answer = predict(question, context, tokenizer=self.tokenizer, model=self.model)

        payload['reader'] = 0
        if len(answer[0]) > 0:
            H, T, img = post_process(html_article, answer, payload['html_url'], tokenizer=self.tokenizer)
            if H != '':
                payload['html_snippet'], payload['text_snippet'], payload['images'] = H, T, img

                payload['reader'] = 1

        return JSONResponse(content=payload)
