import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
# wordcloud==1.8.2.2
# pillow==8.4.0


class DF_wordcloud:    # 自定义的WordCloud类，可以对指定文本进行词频统计以及可视化
    def __init__(self, df):   # 初始化方法(将DataFrame数据作为参数传给类，将df作为类内参数)
        self.df = df   # 参数df表示DataFrame数据框
    def wordcloud_allwords(self):    # 类内方法，统计所有文本中的词频以及可视化
        all_words = WordCloud(       # 创建一个WordCloud对象
            background_color='black',   # 词云图像的参数，包括大小以及背景颜色
            width=2000,
            height=1600
        ).generate(' '.join(self.df['Text']))    # 使用generate(text) 方法从文本数据生成词云
        # （self.df[“Text”]获取数据框的Text列的所有数据，然后使用.join方法将其拼接成一个字符串）
        plt.figure(figsize=(10, 5))
        plt.imshow(all_words, interpolation='bilinear')
        plt.axis('off')  # 不显示坐标轴
        plt.title('All Words', fontsize=20)
        plt.show()
    def wordcloud_positive_words(self):  # 类内方法，统计所有积极标签的文本中的词频以及可视化
        postitive_words = WordCloud(
            background_color='black',
            width=2000,
            height=1600
        ).generate(' '.join(self.df[self.df["Label"]==1.0]['Text']))
        # 先使用self.df["Label"]==1.0获取符合条件的掩码数组，然后再使用self.df[self.df["Label"]==1.0]['Text']获取符合条件的数据框的Text列的数据，最后剑气拼接成字符串
        plt.figure(figsize=(10, 5))
        plt.imshow(postitive_words, interpolation='bilinear')
        plt.axis('off')  # 不显示坐标轴
        plt.title('Postitive Words', fontsize=20)
        plt.show()
    def wordcloud_negative_words(self):   # 类内方法，统计所有消极标签的文本中的词频以及可视化
        negative_words = WordCloud(
            background_color='black',
            width=2000,
            height=1600
        ).generate(' '.join(self.df[self.df["Label"]==0.0]['Text']))
        plt.figure(figsize=(10, 5))
        plt.imshow(negative_words, interpolation='bilinear')
        plt.axis('off')  # 不显示坐标轴
        plt.title('Negative Words', fontsize=20)
        plt.show()

    @classmethod   # 使用@classmethod修饰的是类内方法，可通过类名直接调用（上述的方法是实例方法，必须通过类的实例去调用）
    def show(cls, counting, label_list, Xlabel, Ylabel, Title):   # 定义类内方法，绘制柱状图
        plt.bar(label_list, counting)  # plt.bar函数用于绘制条形图
        # label_list 是条形图的x轴标签，表示不同的情感类别
        # counting 是y轴的值，表示每个情感类别出现的频率计数
        plt.xlabel(Xlabel)  # x轴标题
        plt.ylabel(Ylabel)  # y轴标题
        plt.title(Title)  # 条形图的标题
        plt.show()  # 显示绘制的图表


"""if __name__ == "__main__":
    dw = DF_wordcloud("../Sentiment analysis using transformer/dataframe.csv")
    dw.wordcloud_allwords()
    dw.wordcloud_positive_words()
    dw.wordcloud_negative_words()"""


