# sumandtitle
使用bert-seq2seq-mastter+pytorch+flask+vue实现内容摘要和标题申城
介绍
使用bert-seq2seq-mastter+pytorch+flask+vue实现内容摘要和标题申城

软件架构
web：vue+flask 深度学习:pytorch+bert-seq2seq-mastter 标题生成使用t5model 内容摘要使用：RoBERTa

安装教程
解压 flask 包，然后进入 flask 目录下，命令行输入 pip install -r requirements.txt 安装依赖
进入 flask 目录下命令行输入 python app.py
浏览器输入：http://127.0.0.1:3000/
使用说明
智能创作模块只支持json文件的读入，且每个json文件只能放一个文章内容
文件模板 [ { "id": 1, "content": "文章内容" } ]
可能存在跨域问题待解决

#  所需文件
1、需要自己的训练集
2、最好设置预训练模型
3、可能存在跨域问题待解决
4、模型训练： roberta_auto_title_train.py t5_auto_title_train.py 两个py文件是训练文件 
5、下载相应的训练集和模型，然后进行训练，训练完成就可以进行推理
