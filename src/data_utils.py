import spacy
import json
import jieba
from tqdm import tqdm

# 加载英文语言模型
nlp = spacy.load("en_core_web_sm")


# 由于文件中有多行，直接读取会出现错误，因此一行一行读取
file = open(
    "D:/WorkSpace/dataset/translation2019zh/translation2019zh_train.json",
    "r",
    encoding="utf-8",
)
data = []
for line in file.readlines():
    dic = json.loads(line)
    data.append(dic)


# 分词函数
def tokenize_text(text, lang):
    doc = nlp(text)
    if lang == "en":
        tokens = [token.text for token in doc]
    elif lang == "ch":
        tokens = jieba.lcut(text)
    else:
        raise NotImplementedError
    return tokens


# 对数据集进行分词处理
for entry in tqdm(data, desc="tokenizing"):
    english_tokens = tokenize_text(entry["english"], "en")
    chinese_tokens = tokenize_text(entry["chinese"], "ch")
    # 这里你可以根据你的需求进行后续处理，比如将分词后的结果保存到新的数据结构中或者进行其他操作
    # 例如：entry["english_tokens"] = english_tokens
    #      entry["chinese_tokens"] = chinese_tokens

# 打印示例数据
print("Example English Tokens:", data[0]["english_tokens"])
print("Example Chinese Tokens:", data[0]["chinese_tokens"])
