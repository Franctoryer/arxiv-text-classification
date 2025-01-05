import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import csv
import os
from tqdm import tqdm
import re
import enchant
from dataclasses import dataclass
import spacy
import pytextrank


@dataclass
class Result:
  review_id: str
  label: str
  keyword: str          # 关键词
  keysentence: str          # 关键句
  abstract: str    # 摘要

# 检查拼写
d = enchant.Dict("en_US")

# 初始化词形还原器
lemmatizer = nltk.stem.WordNetLemmatizer()

# 定义停用词
stop_words = list(stopwords.words('english'))
my_stopwords = open('./stopwords.txt', 'r', encoding='utf-8').read().split('\n')
stop_words = [*stop_words, *my_stopwords]

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")


# 分词并去除停用词
def preprocess_text(text):
    # 只保留字符和部分标点符号
    text = re.sub(r'[^a-zA-Z,.\-\+ ]', '', text)
    return text


def get_keyword(document: list[str]) -> str:
  """获取摘要部分"""
  # 使用TfidfVectorizer计算TF-IDF
  vectorizer = TfidfVectorizer(
      stop_words=stop_words,
      decode_error='ignore',
      ngram_range=(1, 1),
      sublinear_tf=True,
      max_df=0.7,
      min_df=5
  )
  tfidf_matrix = vectorizer.fit_transform(document)

  # 获取特征（词）和对应的TF-IDF值
  feature_names = vectorizer.get_feature_names_out()
  tfidf_scores = tfidf_matrix.toarray()[0]

  # 将词和对应的TF-IDF值组合成字典
  tfidf_dict = dict(zip(feature_names, tfidf_scores))

  # 按TF-IDF值排序，提取前N个关键词
  N = 50  # 提取前20个关键词
  top_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
  filter_kws = []
  for kw in top_keywords:
      lemma_kw = lemmatizer.lemmatize(kw[0])
      # 进行词性标注
      pos_tags = nltk.pos_tag([lemma_kw])
      pos = pos_tags[0][1]
      if lemma_kw not in filter_kws and len(kw[0]) >= 4 and d.check(kw[0]) and (pos.startswith('NN') or pos.startswith('VB')):
          filter_kws.append(lemma_kw)

  kws = ' '.join(filter_kws[:N])

  return kws

def get_abstract(document: str) -> str:
  """获取摘要部分"""
  document = document[:4000]
  pattern = r"a\s?bstract\s+([\s\S]*?)\n\n"
  abstract = re.findall(pattern, document)
  if len(abstract) >= 1:
    abstract = abstract[0].replace('\n', ' ')
    return abstract
  else:
    return ""
  
def get_key_sentence(document: str) -> str:
  doc = nlp(document)
  # examine the top-ranked sentences in the document
  results = []
  for sent in doc._.textrank.summary(limit_phrases=10, limit_sentences=5):
    results.append(sent.text.replace('\n', ' '))
  
  return ' '.join(results)


with open('./result.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['review_id', 'label', 'keywords', 'keysentences', 'abstract'])

labels = os.listdir('../datasets/')
for label in labels:
    dir_name = f'../datasets/{label}'
    files = os.listdir(dir_name)
    sum = 0
    for file in tqdm(files):
      try:
        # 每个标签只要1500样本
        if sum >= 1500:
            break
        document = open(f'{dir_name}/{file}', 'r', encoding='utf-8').read().lower()

        # 摘要部分
        abstract = get_abstract(document)
        if len(abstract) <= 100:
            continue
        # 预处理文本
        processed_text = preprocess_text(document)
        # 关键词部分
        kws = get_keyword(sent_tokenize(processed_text))
        # 关键句部分
        key_sentences = get_key_sentence(processed_text)

        # 保存结果
        result = Result(file, label, kws, key_sentences, abstract)
        with open('./result.csv', 'a+', encoding='utf-8-sig', newline='') as f:
          writer = csv.writer(f)
          writer.writerow([file, label, kws, key_sentences, abstract])
          sum += 1
      except Exception as e:
         continue
    

# if __name__ == '__main__':
#     document = open('../datasets/cs.AI/0103009v3.pdf.txt', 'r', encoding='utf-8').read().lower()
#     get_abstract(document)