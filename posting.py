class Posting:
    def __init__(self, doc_id, score):
        self.document_id = doc_id
        self.tfidf_score = score

    def __str__(self):
        return f'"{self.document_id}": {self.tfidf_score}'

    def __repr__(self):
        return f"id={self.document_id}: score={self.tfidf_score}"
