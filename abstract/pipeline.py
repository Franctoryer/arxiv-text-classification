from transformers import pipeline
from nltk.tokenize import RegexpTokenizer
import os
# import nltk
# nltk.download('punkt')  # 下载用于分词的预训练模型
# nltk.download('averaged_perceptron_tagger')  # 如果需要词性标注
# nltk.download('maxent_ne_chunker')  # 如果需要命名实体识别
# nltk.download('words')  # 如果需要使用内置的词汇表

# 加载摘要生成器
summarizer = pipeline("summarization", model="allenai/led-base-16384", tokenizer="allenai/led-base-16384")
# 定义正则表达式分词器，只保留字母和数字
tokenizer = RegexpTokenizer(r'[\w]+|[^\w\s]')

labels = os.listdir('../datasets/')
for label in labels:
  dir_name = f'../datasets/{label}'
  files = os.listdir(dir_name)
  files = files[:1500]
  for file in files:
    document = open(f'{dir_name}/{file}', 'r', encoding='utf-8').read()
    
    # 文本分词
    words = tokenizer.tokenize(document)
    document = ' '.join(words[:8000])
    try:
      # 生成摘要
      summary = summarizer(document, max_length=400, min_length=100, do_sample=True)
      # 输出摘要
      abstract = summary[0]["summary_text"]
    except Exception as e:
      pass

    print(summary, label)
