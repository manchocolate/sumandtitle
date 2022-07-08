from flask_cors import *
import torch
from flask import Flask,request,render_template,send_from_directory
from bert_seq2seq.tokenizer import  load_chinese_base_vocab, T5PegasusTokenizer
import json
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
from bert_seq2seq.t5_ch import T5Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = "./state_dict/t5-chinese/vocab.txt"
model_path = "./state_dict/t5_autotile.bin"
word2idx = load_chinese_base_vocab(vocab_path)
model = T5Model(word2idx, size="base")
model.set_device("cpu")
model.load_all_params(model_path)
model.eval()

auto_title_model = "./state_dict/bert_auto_title_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = "./state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
model_name = "roberta"  # 选择模型名字
# 加载字典
word2idx = load_chinese_base_vocab(vocab_path)
# 定义模型
bert_model = load_bert(word2idx, model_name=model_name)
bert_model.set_device(device)
bert_model.eval()
## 加载训练的模型参数～
bert_model.load_all_params(model_path=auto_title_model, device=device)
# for t in all_txt:
app = Flask(__name__,
            template_folder="./web/dist",
            static_folder="./web/dist",
            static_url_path="")

CORS(app, supports_credentials=True)





@app.route('/')
def index():
    if request.method == 'GET':
        return render_template("index.html")




@app.route("/upload/post",methods=['POST'])
def post():
    """接受前端传送来的文件"""
    file_obj = request.files.get("file")
    data = json.load(file_obj)
    for item in data:
        content = item['content']
        title = model.sample_generate_encoder_decoder(content)
        abstract = bert_model.generate(content)
        print(title)
    t = {
        "title": title,
        "abstract": abstract
    }
    return json.dumps(t)
@app.route("/upload/post2")
def post2():
    t = {

        "title": "dddd",
        "abstract": "titi"

    }
    return json.dumps(t)


if __name__ == '__main__':
    app.run(debug=True, port=3000)