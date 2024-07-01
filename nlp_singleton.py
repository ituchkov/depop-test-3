import spacy


class NLPModelSingleton:
    _instance = None
    _loaded = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self.load_model()
            self._loaded = True

    def __call__(self, text: str):
        return self.nlp(text)

    def load_model(self):
        self.nlp = spacy.load("en_core_web_sm")

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
