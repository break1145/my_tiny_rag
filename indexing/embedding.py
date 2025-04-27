import json
import re

from openai import OpenAI

from OpenaiModel import OpenAIModel
from VectorStore import PineconeVS


class BaseEmbedding:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def text_embedding(self, content):
        raise NotImplementedError()

    def image_embedding(self, content):
        raise NotImplementedError()

    def excel_embedding(self, content):
        raise NotImplementedError()

    def pdf_embedding(self, content):
        raise NotImplementedError()


class PineconeEmbedding(BaseEmbedding):
    def __init__(self, vectorstore):
        super().__init__(vectorstore)
        self.openai_model = OpenAIModel()

    def text_embedding(self, content):
        """
        对长文本进行简单分割，按段落embedding
        :param content: 需要分割的文本
        :return: 一个List，包含分割后文本的向量
        """
        vectors = []
        # 按句子分割
        vectors += self.openai_model.get_embedding_batch(
            split_by_sentence(content)
        )
        # 按段落分割
        vectors += self.openai_model.get_embedding_batch(
            split_by_paragraph(content)
        )
        # todo: 接入其他分割方式
        return vectors




def split_by_sentence(content):
    sentences = re.split(r'(?<=[.!?。！？])\s*', content)
    return [s.strip() for s in sentences if s.strip()]
def split_by_paragraph(content):
    paragraphs = re.split(r'\n+', content)
    return [p.strip() for p in paragraphs if p.strip()]

#
# if __name__ == '__main__':
#     text = """
#     游戏是基于被遗忘的国度设定集中的设定而设计的，使用AD&D 2e，即高级龙与地下城 2版规则，尽管规则中的各种参数已被作特定的修改以适应游戏的实时战斗模式。而游戏可以随时暂停，每个角色也都可以完成连续而不因暂停而中断的动作。
#
#     在游戏中，过去和现在的事件通过对话、书面文本、日记条目或过场动画与玩家相关联。当玩家点击一个角色或NPC时，将开始一段对话。具有可选语言选项的书面对话，有时还会生成语音。对话有时会导向不同的任务或事件。
#
#     《博德之门 I》共分为七个章节，其中穿插着旁白语言和过场动画。每章都允许玩家自由探索世界地图，但有些地点直到前进到游戏的某个具体节点才会解锁。玩家开始时相当弱小，装备简陋，没有队友。但在游玩过程中可以获得新的更强大的武器、护甲和法术，并且可以组成一个最多由六名同伴（包括玩家角色）组成的队伍。通过完成任务和杀死怪物获得的经验值可以提高主角和同伴的能力。
#
#     游戏中的时间流逝表现为光线即屏幕亮度的变化、大多数商店的开门和打烊以及夜间遭遇伏击的可能性增加。酒馆和旅店在夜间也营业，但顾客或老板不会时间的流逝而变化。玩家控制的队伍有时在旅行一整天后会感到疲劳，需要休息才能恢复。
#
#     """
#     # print(split_by_paragraph(text))
#     # print(len(split_by_paragraph(text)))
#     #
#     # print(split_by_sentence(text))
#     # print(len(split_by_sentence(text)))
#
#     emb = PineconeEmbedding(vectorstore=PineconeVS('test_1'))
#     vectors = emb.text_embedding(text)
#     print(len(vectors[0]['values']))