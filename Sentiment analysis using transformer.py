import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Data_Prerocessing_Function import nltk_download, DataCleaning, lemmatization, TextDataset
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import warnings
from train_and_test import Train_And_Test
from word_cloud import DF_wordcloud as DW
warnings.filterwarnings("ignore")     # 忽略警告信息
nltk_download('wordnet')    # 使用nltk_download函数下载wordnet数据集
nltk_download('wordnet2022')    # 下载wordnet2022数据集
nltk_download('omw-1.4')


def tokenize_text(row, max_length):    # 分词器函数，将token序列转换为对应的数值化token序列，
    # 参数为一行数据和最大长度
    text = row['Text']    # 先提取该行的Text列的数据
    tokenized = tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=max_length)
    # 使用encod_plus函数对该文本进行token数值化，并且填充到最大长度，返回值为一个包含多个键值对的字典
    return tokenized['input_ids']    # 将字典中input_ids对应的值（token的数值化序列）返回


if __name__ =="__main__":
    # 读取数据集
    youtube_comments = pd.read_csv('../Sentiment analysis using transformer/dataset/Youtube Statistics/comments.csv')
    sentiment_tweets = pd.read_csv('../Sentiment analysis using transformer/dataset/Sentiment Dataset with 1 Million Tweets/dataset.csv')
    print(youtube_comments.head())   # 输出查前五条数据
    print(sentiment_tweets.head())

    # 对youtube的数据集进行预处理
    # 可视化youtube数据集
    counting = youtube_comments.Sentiment.value_counts()    # 返回一个pandas的series对象
    # 统计youtube_comments数据集中Sentiment列的值出现的频率
    """plt.bar(['Positive', 'Neutral', 'Negative'], counting)   # plt.bar函数用于绘制条形图
    # ['Positive', 'Neutral', 'Negative'] 是条形图的x轴标签，表示不同的情感类别
    # counting 是y轴的值，表示每个情感类别出现的频率计数
    plt.xlabel('The type of sentiment')    # x轴标题
    plt.ylabel('Number of the Samples')    # y轴标题
    plt.title("The Number of Samples for each sentiment")    # 条形图的标题
    plt.show()    # 显示绘制的图表"""
    DW.show(counting, ['Positive', 'Neutral', 'Negative'], 'The type of sentiment', 'Number of the Samples',
         "The Number of Samples for each sentiment")
    First_corpus = pd.DataFrame(columns=['Text', 'Label'])   # 创建一个空的pandas DataFrame（数据框），该DataFrame包含两列:’Text‘和’Label‘
    First_corpus['Text'], First_corpus['Label'] = youtube_comments['Comment'], youtube_comments['Sentiment']
    # 分别提取youtube_comments的Comment和Sentiment列的数据，将其赋给新创建的DataFrame
    # 去除中性情绪（Label=1.0）的行，将积极情绪对应的label的值改为1.0
    First_corpus.drop(First_corpus[First_corpus['Label'] == 1.0].index, inplace=True)
    # 使用drop函数删除Label列里值为1.0的行，inplace=True表示直接在原数据框上操作
    First_corpus.loc[First_corpus['Label'] == 2.0, 'Label'] = 1.0
    # 先使用First_corpus['Label'] == 2.0获取Label列为2.0的一个布尔series（数组，满足条件的数据框的索引对应的下标为true，反之为false）
    # loc[First_corpus['Label'] == 2.0, 'Label'] = 1.0将满足条件的行的Label列的值改为1.0
    print(First_corpus.head(10))   # 输出查看Firdt_croups数据框的前10行
    print(First_corpus.Text.describe())   # 输出查看First_croups数据框里的Text列的统计信息
    # describe() 方法会返回以下几个统计信息：
    # count：非空值的数量。 unique：唯一值的数量。 top：出现次数最多的值。 freq：最频繁值的出现次数
    First_corpus.drop_duplicates(inplace=True)    # drop_duplicates函数去除First_croups数据框里的重复值（行）
    print(First_corpus.Text.is_unique)   # First_corpus.Text.is_unique判断数据框里的Text列的值是否唯一，若唯一，则返回true，反之则返回false
    print(f'Number of Samples after removing duplication: {len(First_corpus)}')  # 查看去重后的样本数量
    First_corpus.dropna(inplace=True)    # dropna函数去除数据框里包含缺失值的行
    print(f'Number of Samples after removing null values: {len(First_corpus)}')

    # 对tweets的数据集进行预处理
    print(sentiment_tweets.head())
    print(f"The number of samples before dropping Non-English texts: {len(sentiment_tweets)}")
    sentiment_tweets = sentiment_tweets[sentiment_tweets['Language'] == 'en']
    # 只保留sentiment_tweets数据框里的Lan列为en的行(查看可知，tweets数据集中的数据不只包含english语言，要求只包含english语言的数据)
    print(f"The number of samples AFTER dropping Non-English texts: {len(sentiment_tweets)}")
    # 可视化tweets数据集
    counting = sentiment_tweets.Label.value_counts()   # 统计sentiment_tweets数据集中各列的值的出现次数
    """plt.bar(['Positive', 'Negative', 'Uncertainty', 'Litigious'], counting)
    plt.xlabel('The type of sentiment')
    plt.ylabel('Number of the Samples')
    plt.title("The Number of Samples for each sentiment")
    plt.show()"""
    DW.show(counting, ['Positive', 'Negative', 'Uncertainty', 'Litigious'], 'The type of sentiment', 'Number of the Samples',
        "The Number of Samples for each sentiment")   # 通过类名直接调用类内方法

    Second_corpus = pd.DataFrame(columns=['Text', 'Label'])  # 创建一个新的数据框Second_croups，包含Text和Label两列
    sentiment_tweets.loc[sentiment_tweets['Label'] == 'positive', 'Label'] = 1.0
    # 将sentiment_tweets数据集中Label列为positive和negative的列对应的；abel值替换为1.0和2.0（与First_croups一致，方便后续的数据合并）
    sentiment_tweets.loc[sentiment_tweets['Label'] == 'negative', 'Label'] = 0.0
    sentiment_tweets.drop(sentiment_tweets[(sentiment_tweets['Label'] == 'uncertainty') | (
                sentiment_tweets['Label'] == 'litigious')].index
                          , inplace=True)   # 去除sentiment_tweets数据集里label列值为uncertainty和litigious的行
    Second_corpus['Text'], Second_corpus['Label'] = sentiment_tweets['Text'], sentiment_tweets['Label']
    # 将sentiment_tweets数据集里Text列和Label列的值分别赋给数据框Second_croups的Text列和Label列
    Second_corpus.drop_duplicates(inplace=True)  # 去除数据框Second_croups的重复行
    Second_corpus.Text.isna().value_counts()  # 返回数据框里Text列的缺失值的情况，
    Second_corpus.dropna(inplace=True)  # dropna函数去除数据框里包含缺失值的行

    # 合并两个数据框
    df = pd.concat([First_corpus, Second_corpus], ignore_index=True)
    # 使用concat函数合并两个DataFrame（数据框），ignore_index=True 表示重置索引。合并后的DataFrame索引将从0到n-1重新排列（n是合并后行的总数）
    print(df.Label.value_counts())   # 输出查看数据框的标签分布
    print(f'The number of total samples after concatenating: {len(df)}')    # 输出查看合并后的数据框的数据量
    print(df.head(10))
    labels = df.groupby(df['Label']).size().index    # 将数据框的数据按照Label列的值进行分类
    # .size() 计算每个组的大小，即每个标签的样本数量。
    # .index 获取分组后的标签值。
    # .values 获取每个标签对应的样本数量
    # values = df.groupby(df['Label']).size().values
    # fig = go.Figure(
        # data=[go.Pie(labels=labels, values=values, hole=.6, title='Distribution of Classes in the dataset')])  # go.Figure(data=[...]) 创建一个 Figure 对象，其中包含一个 Pie（饼图）对象
    # go.Pie()定义饼图的属性
    # labels=labels 指定饼图的标签，即不同类别的标签值。
    # values=values 指定饼图的数值，即每个标签的样本数量。
    # hole=.6 创建一个带有中间空洞的环形图，空洞大小为 60%。
    # title='Distribution of Classes in the dataset' 指定饼图的标题
    # fig.show()   #  显示图表

    # 数据清洗
    df.drop_duplicates(inplace=True)   # 去除重复数据
    df.dropna(inplace=True)            # 去除包含缺失值的数据
    df = DataCleaning(df)    # 调用数据框清洗函数
    df.drop_duplicates(inplace=True)   # 再次去除重复数据和包含缺失值的数据（防止数据框清晰过程引入新的错误）
    df.dropna(inplace=True)
    dw = DW(df)     # 实例化DW对象，使用词云进行词频统计（所有单词、消极标签的单词、积极标签的单词）
    dw.wordcloud_allwords()
    dw.wordcloud_positive_words()
    dw.wordcloud_negative_words()
    # 数据处理
    df['Text'] = df['Text'].apply(remove_stopwords)    # 对数据框里的Text列的数据应用remove_stopwords函数，即去除停用词
    df['Text'] = df['Text'].apply(lemmatization)       # 对数据框里的Text列的数据应用lemmatization函数，即词形还原
    # file_path = "../Sentiment analysis using transformer/dataframe.csv"  # 将dataframe数据保存为csv文件
    # df.to_csv(file_path, index=False)     # 参数index=False表示在csv文件里不包含dataframe的索引
    # print("csv文件保存成功")
    # df = pd.read_csv("../Sentiment analysis using transformer/dataframe.csv")
    # Tokenization（标记化）
    model_ckpt = "bert-base-uncased"  # 变量model_ckpt存放模型bert-base-uncased
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)  # 从预训练模型中加载分词器，加载完成后，tokenizer对象可以用于对文本进行编码、解码以及生成模型的输入
    # rom_pretrained用于加载训练预模型，AutoTokenizer类可以自动根据给定的模型名称或路径选择合适的分词器
    # 标记化的例子
    """text = df['Text'][0]   # 提取数据框里的Text列的第一条数据
    print("\nOur text to tokenize:", text, "\n")  # 输出要标记化的文本

    tokenized_text = {}   # 字典存放标记化结果
    max_length = 50  # 设置文本的最大长度（即最大token数）
    encoded = tokenizer.encode_plus(  # 使用 tokenizer.encode_plus 对文本进行标记化
        # encode_plus 方法返回一个字典，包含 input_ids（数值token）、attention_mask 等信息
        text,
        # add_special_tokens=True,  # 添加特殊token [CLS] 和 [SEP]
        padding='max_length',   # 填充到最大长度
        truncation=True,       # 参数用于控制当输入文本长度超过指定的最大长度时，是否截断输入文本
        max_length=max_length  # 指定最大长度
    )
    tokenized_text['Numerical Token'] = encoded['input_ids']  # tokenizer.encode_plus函数返回的字典里的input_ids对应的值即为标记后的数值Token
    # 将标记后的数值Token和Numerical Token组成键值对放入字典tokenized_text里
    tokenized_text['Token'] = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    # 将数值token转换回对应的token，并存储在字典 tokenized_text 中，键为 'Token'

    print("Tokenizer has a vocabulary size of", tokenizer.vocab_size, "words.")   # 输出分词器（即tokenizer）的词汇表的大小（即分词器可识别的词汇数量）
    print(pd.DataFrame(tokenized_text).T)   # tokenized_text字典里数据存放形式为：每行为一个Token与其对应的数值，所以要转置一下，使得一段文本的token在一行，对应的数值token在一行
    # 例如：转置后的输出如下：
    #                         0       1       2       3        4         5      6      7
    # Numerical Token       101    2023    2003    2019    28307      6251   1012    102
    # Token               [CLS]    this      is      an  example  sentence      .  [SEP]"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU，有则设置为gpu，否则设置为cpu
    # Making a lightweight config class that allows for struct like attribute accessing
    class Config:  # 定义一个 Config 类，用于将配置字典转换为对象的属性。
        def __init__(self,
                     config_dict):  # 初始化方法，接收一个字典 config_dict 并使用 self.__dict__.update(config_dict) 更新对象的属性，使得我们可以像访问对象属性一样访问配置参数
            self.__dict__.update(config_dict)
    # 设置模型参数字典
    config = {
        'vocab_size': tokenizer.vocab_size,  # 设置词汇表的大小为分词器的词汇表的大小
        'embedding_dimensions': 256,  # 设置嵌入层的维度
        'max_tokens': 50,  # 设置每个序列的最大token数
        'num_attention_heads': 16,  # 设置多头注意力的头数（即多头注意力中有多少个独立的注意力机制）
        'hidden_dropout_prob': 0.5,  # 设置 Dropout 概率，用于防止过拟合，表示有50%的概率将某些神经元的输出置零
        'intermediate_size': 256 * 4,  # 设置中间隐藏层的神经元数量，通常是嵌入层维度的四倍
        'num_encoder_layers': 2,  # 设置编码器的数量
        'device': device  # 设置计算设备
    }

    config = Config(config)  # 使用 Config 类将配置字典 config 包装成一个 Config 对象。
    # 通过 Config(config) 创建的 config 对象，可以像访问对象属性一样访问配置参数。例如，可以使用 config.vocab_size 来获取词汇表大小

    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)  # 使用train_test_split函数将数据集df分隔为训练集和验证集
    # test_size=0.3表示训练集占df的70%，验证集val占30%，random_state=42设置随机数种子，保证每次的分割结果一致（42是一个常用的随机数种子）
    val_df, test_df = train_test_split(val_df, test_size=0.4, random_state=42)
    print(train_df.head())
    # 将验证集val分为验证集和测试集

    max_length = config.max_tokens  # 设置文本的最大长度（即最大token数）

    train_df['tokenized'] = train_df.apply(lambda row: tokenize_text(row, max_length), axis=1)
    # 对训练集的每一行数据都应用分词器函数tokenize_text
    val_df['tokenized'] = val_df.apply(lambda row: tokenize_text(row, max_length), axis=1)
    # 对验证集的每一行数据都应用分词器函数tokenize_text
    test_df['tokenized'] = test_df.apply(lambda row: tokenize_text(row, max_length), axis=1)
    # 对测试集的每一行数据都应用分词器函数tokenize_text
    print(train_df.head())
    print(train_df['tokenized'].head(10))
    df['Text_length'] = train_df['tokenized'].apply(lambda x: len(x))
    print(df["Text_length"].head(100))

    X_train = train_df['tokenized']  # 将先前分好的训练数据的tokenized列的数据（数值化token序列），赋给X_train
    y_train = train_df['Label']  # 将训练数据的Label列的数据（标签）赋给y_train
    # X_train存放训练集的数值化Token序列，y_train存放数据对应的标签
    X_val = val_df['tokenized']
    y_val = val_df['Label']
    # X_val存放验证集的数值化Token序列，y_val存放数据对应的标签
    X_test = test_df['tokenized']
    y_test = test_df['Label']
    # X_test存放测试集的数值化Token序列，y_test存放数据对应的标签
    # Create both datasets
    train_dataset = TextDataset(X_train, y_train)  # 使用TextDataset（）函数创建训练数据集
    val_dataset = TextDataset(X_val, y_val)  # 使用TextDataset（）函数创建验证数据集
    test_dataset = TextDataset(X_test, y_test)  # 使用TextDataset（）函数创建测试数据集

    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=256,
                                  shuffle=True)  # Shuffle for random sampling without replacement
    # 创建了一个 PyTorch 的 DataLoader 对象（数据加载器），用于从 train_dataset 中按批次加载数据，并在每个 epoch 期间随机打乱数据
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    # 调用模型训练和测试函数
    Train_And_Test(config, train_dataloader, val_dataloader, test_dataloader, 200)


