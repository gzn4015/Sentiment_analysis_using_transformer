import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset
import torch


def nltk_download(dir):     # 函数功能:下载数据集并存储到本地
    str1 = "D:/Python/pythonProject2/Sentiment analysis using transformer/"    # 该项目的本地地址
    data_dir = str(str1 + dir)   # 拼接字符串，得到数据集的存储地址
    if not os.path.exists(data_dir):   # 不存在文件，则创建
        os.makedirs(data_dir)

    # 将路径添加到 nltk 数据路径列表中
    nltk.data.path.append(data_dir)
    # 下载 WordNet 数据集到指定路径
    nltk.download(dir, download_dir=data_dir)


def cleaning(text):    # 文本清洗函数，使用正则表达式去除文本里不需要的字符
    text = re.sub(r'#\w+', '', text)                 # Removing Hashtags
    text = re.sub(r'http\S+', '', text)              # Removing Links & URLs
    text = re.sub(r'@\w+', '', text)                 # Removing Mentions
    text = re.sub('[()!?.\';:<>`$%’,]', '', text)   # Removing Punctuations with different forms
    text = re.sub(r'[^a-zA-Z]', ' ', text)           # Removing digits
    text = re.sub(r'([a-zA-Z])\1{2,}', '\1', text)   # Reduce duplicated character (> 3) to only one
    return text


def DataCleaning(corpus):    # 数据框清洗函数
    abbreviations = {'fyi': 'for your information',  # 定义一个字符串替换字典，将数据中的一些缩写替换为完整的句子
                     'lol': 'laugh out loud',
                     'loza': 'laughs out loud',
                     'lmao': 'laughing',
                     'rofl': 'rolling on the floor laughing',
                     'vbg': 'very big grin',
                     'xoxo': 'hugs and kisses',
                     'xo': 'hugs and kisses',
                     'brb': 'be right back',
                     'tyt': 'take your time',
                     'thx': 'thanks',
                     'abt': 'about',
                     'bf': 'best friend',
                     'diy': 'do it yourself',
                     'faq': 'frequently asked questions',
                     'fb': 'facebook',
                     'idk': 'i don\'t know',
                     'asap': 'as soon as possible',
                     'syl': 'see you later',
                     'nvm': 'never mind',
                     'frfr': 'for real for real',
                     'istg': 'i swear to god',
                     }
    corpus['Text'] = corpus['Text'].apply(cleaning)   # 对数据框里的Text列的所有数据都调用cleaning（）函数
    # apply() 方法用于对 DataFrame 中的每个元素应用指定的函数
    corpus['Text'] = corpus['Text'].str.lower()   # 将数据框里的Text列的数据都转换为小写
    for abbreviation, full_form in abbreviations.items():   # 循环获取字符串替换字典里的所有键值对（abbreviation表示缩写，full_form表示全称）
        corpus['Text'] = corpus['Text'].str.replace(abbreviation, full_form)   # 对于数据框里的Text列的所有数据都应用replace函数，将缩写替换为对应的全称
    return corpus


def lemmatization(sentence):   # 词形还原函数
    lemmatizer = WordNetLemmatizer()  # 创建一个WordNetLemmatizer实例对象，用于之后的词形还原操作（词形还原是指将单词还原到他的基本形式）
    # 词形还原的例子
    # sentence = "The cats are running"
    # lemmatized_sentence = lemmatization(sentence)
    # print(lemmatized_sentence)
    # 输出结果为The cat are running（WordNetLemmatizer默认将单词视为名词，所以没有对running进行词形还原）
    words = sentence.split()  # 将句子中的单词以空格分隔开
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # 列表推导式将对 words 中的每个单词应用 lemmatizer.lemmatize() 方法，并将结果存储在 lemmatized_words 列表中
    return " ".join(lemmatized_words)  # 将列表中的元素拼接成字符串返回


class TextDataset(Dataset):        # 继承于父类Dataset，用于创建训练集、验证集、测试集以及将他们加载到DataLoader里，以便在模型里对其进行批处理
    def __init__(self, x_dataframe, y_dataframe):   # 初始化方法，接受两个参数，序列数据以及其对应的标签
        self.x_dataframe = x_dataframe
        self.y_dataframe = y_dataframe

    def __len__(self):      # 类内的特殊方法，可以实现使用len(类对象)直接得到数据集的样本数量
        return len(self.x_dataframe)

    def __getitem__(self, idx):   # 类内特殊方法，可以实现根据索引直接返回数值化token序列以及其对应的标签
        # 使用示例: x、y=train_dataset[0],即x表示数据集train_dataset的di1个元素的数值化token数据，y表示其对应的标签
        x = self.x_dataframe.iloc[idx]  # Get the 'tokenized' data
        y = self.y_dataframe.iloc[idx]  # Get the 'target' data
        return torch.LongTensor(x), torch.tensor(y, dtype=torch.float32)
        # 将x转为pytorch长整型张量，将y转为pytorch浮点型张量返回（PyTorch 的模型和许多操作都期望输入是 PyTorch 张量）