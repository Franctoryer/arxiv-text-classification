import spacy
import pytextrank

# example text
text = open('../datasets/cs.AI/0103009v3.pdf.txt', 'r', encoding='utf-8').read().lower()

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")

doc = nlp(text)

# examine the top-ranked sentences in the document
for sent in doc._.textrank.summary(limit_phrases=10, limit_sentences=5):
    print(sent.text.replace('\n', ' '))