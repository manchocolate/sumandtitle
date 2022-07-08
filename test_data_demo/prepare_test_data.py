import torch
from flask import Flask
from bert_seq2seq.tokenizer import  load_chinese_base_vocab, T5PegasusTokenizer
from bert_seq2seq.extend_model_method import ExtendModel
import glob
import json
import socket
import py_eureka_client.eureka_client as eureka_client
import py_eureka_client.netint_utils as netint_utils
from bert_seq2seq.t5_ch import T5Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = "./state_dict/t5-chinese/vocab.txt"
model_path = "./state_dict/t5_autotile.bin"
word2idx = load_chinese_base_vocab(vocab_path)

model = T5Model(word2idx, size="base")
model.set_device("cpu")
model.load_all_params(model_path)
model.eval()

def read_and_write_data(input_file, output_file):

    fw = open(output_file, 'a', encoding='utf-8')

    test_with_title = []

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            idx = item['id']
            content = item['content']
            title = model.sample_generate_encoder_decoder('text')  # 你的模型预测title
            test_with_title.append({"id": idx, "title": title, "content": content})

    jst = json.dumps(test_with_title, ensure_ascii=False)
    fw.write(jst)


if __name__ == '__main__':
    # 官方发布的测试数据
    input_file = './test_no_title.json'

    # 提交的测试集结果
    # 邮件只上传   团队编号_队伍名称_队长姓名.json文件
    output_file ='./团队编号_队伍名称_队长姓名.json'

    read_and_write_data(input_file, output_file)


