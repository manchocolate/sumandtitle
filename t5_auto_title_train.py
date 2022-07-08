# -*- coding:utf-8 -*-
import torch
import time
import glob
import jieba
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq import T5Model
from tqdm import tqdm 

src_dir = './corpus/auto_title/content.src'
tgt_dir = './corpus/auto_title/title.tgt'

vocab_path = "./state_dict/t5-chinese/vocab.txt" ## 字典
model_path = "./state_dict/t5-chinese/pytorch_model.bin" ## 预训练参数

model_save_path = "./state_dict/t5_autotile.bin" ## 训练完模型 保存在哪里
batch_size = 1
lr = 1e-5
word2idx = load_chinese_base_vocab(vocab_path)
tokenizer = T5PegasusTokenizer(word2idx)

def read_file(src_dir, tgt_dir):
    src = []
    tgt = []

    with open(src_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())

    return src, tgt


class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt):
        ## 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids_src, _ = tokenizer.encode(src, max_length=256)
        token_ids_tgt, _ = tokenizer.encode(tgt, max_length=256)
        output = {
            "token_ids_src": token_ids_src,
            "token_ids_tgt": token_ids_tgt,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids_src = [data["token_ids_src"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt = [data["token_ids_tgt"] for data in batch]
    max_length_tgt = max([len(t) for t in token_ids_tgt])

    token_ids_padded = padding(token_ids_src, max_length_src)
    target_ids_padded = padding(token_ids_tgt, max_length_tgt)
    labels_ids = target_ids_padded.clone()
    labels_ids[labels_ids == 0] = -100
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return token_ids_padded, target_ids_padded, labels_ids


class Trainer:
    def __init__(self):
        # 加载数据
        self.sents_src, self.sents_tgt = read_file(src_dir, tgt_dir)

        # 判断是否有可用GPU
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = T5Model(word2idx)
        ## 加载预训练的模型参数～
        # self.model.load_pretrain_params(model_path)
        self.model.load_all_params(model_save_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = SeqDataset(self.sents_src, self.sents_tgt)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        self.model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        report_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        for token_ids, target_ids, labels_ids in tqdm(dataloader, total=len(dataloader)):
            step += 1
            if step % 100 == 0:
                self.save(model_save_path)
                self.model.eval()

                test_data = ["近期，美国国会众院通过法案，重申美国对台湾的承诺。对此，中国外交部发言人表示，有关法案严重违反一个中国原则和中美三个联合公报规定，粗暴干涉中国内政，中方对此坚决反对并已向美方提出严正交涉。事实上，中国高度关注美国国内打“台湾牌”、挑战一中原则的危险动向。近年来，作为“亲台”势力大本营的美国国会动作不断，先后通过“与台湾交往法”“亚洲再保证倡议法”等一系列“挺台”法案，“2019财年国防授权法案”也多处触及台湾问题。今年3月，美参院亲台议员再抛“台湾保证法”草案。众院议员继而在4月提出众院版的草案并在近期通过。上述法案的核心目标是强化美台关系，并将台作为美“印太战略”的重要伙伴。同时，“亲台”议员还有意制造事端。今年2月，5名共和党参议员致信众议院议长，促其邀请台湾地区领导人在国会上发表讲话。这一动议显然有悖于美国与台湾的非官方关系，其用心是实质性改变美台关系定位。上述动向出现并非偶然。在中美建交40周年之际，两国关系摩擦加剧，所谓“中国威胁论”再次沉渣泛起。美国对华认知出现严重偏差，对华政策中负面因素上升，保守人士甚至成立了“当前中国威胁委员会”。在此背景下，美国将台海关系作为战略抓手，通过打“台湾牌”在双边关系中增加筹码。特朗普就任后，国会对总统外交政策的约束力和塑造力加强。其实国会推动通过涉台法案对行政部门不具约束力，美政府在2018年并未提升美台官员互访级别，美军舰也没有“访问”台湾港口，保持着某种克制。但从美总统签署国会通过的法案可以看出，国会对外交产生了影响。立法也为政府对台政策提供更大空间。然而，美国需要认真衡量打“台湾牌”成本。首先是美国应对危机的代价。美方官员和学者已明确发出警告，美国卷入台湾问题得不偿失。美国学者曾在媒体发文指出，如果台海爆发危机，美国可能需要“援助”台湾，进而导致新的冷战乃至与中国大陆的冲突。但如果美国让台湾自己面对，则有损美国的信誉，影响美盟友对同盟关系的支持。其次是对中美关系的危害。历史证明，中美合则两利、斗则两伤。中美关系是当今世界最重要的双边关系之一，保持中美关系的稳定发展，不仅符合两国和两国人民的根本利益，也是国际社会的普遍期待。美国蓄意挑战台湾问题的底线，加剧中美关系的复杂性和不确定性，损害两国在重要领域合作，损人又害己。美国打“台湾牌”是一场危险的赌博。台湾问题是中国核心利益，中国政府和人民决不会对此坐视不理。中国敦促美方恪守一个中国原则和中美三个联合公报规定，阻止美国会审议推进有关法案，妥善处理涉台问题。美国悬崖勒马，才是明智之举。（作者系中国国际问题研究院国际战略研究所副所长）",
                 "在推进“双一流”高校建设进程中，我们要紧紧围绕为党育人、为国育才，找准问题、破解难题，以一流意识和担当精神，大力推进高校的治理能力建设。增强政治引领力。坚持党对高校工作的全面领导，始终把政治建设摆在首位，增强校党委的政治领导力，全面推进党的建设各项工作。落实立德树人根本任务，把培养社会主义建设者和接班人放在中心位置。紧紧抓住思想政治工作这条生命线，全面加强师生思想政治工作，推进“三全育人”综合改革，将思想政治工作贯穿学校教育管理服务全过程，努力让学生成为德才兼备、全面发展的人才。提升人才聚集力。人才是创新的核心要素，创新驱动本质上是人才驱动。要坚持引育并举，建立绿色通道，探索知名专家举荐制，完善“一事一议”支持机制。在大力支持自然科学人才队伍建设的同时，实施哲学社会科学人才工程。立足实际，在条件成熟的学院探索“一院一策”改革。创新科研组织形式，为人才成长创设空间，建设更加崇尚学术、更加追求卓越、更加关爱学生、更加担当有为的学术共同体。培养学生竞争力。遵循学生成长成才的规律培育人才，着力培养具有国际竞争力的拔尖创新人才和各类专门人才，使优势学科、优秀教师、优质资源、优良环境围绕立德树人的根本任务配置。淘汰“水课”，打造“金课”，全力打造世界一流本科教育。深入推进研究生教育综合改革，加强事关国家重大战略的高精尖急缺人才培养，建设具有国际竞争力的研究生教育。激发科技创新力。在国家急需发展的领域挑大梁，就要更加聚焦科技前沿和国家需求，狠抓平台建设，包括加快牵头“武汉光源”建设步伐，积极参与国家实验室建设，建立校级大型科研仪器设备共享平台。关键核心技术领域“卡脖子”问题，归根结底是基础科学研究薄弱。要加大基础研究的支持力度，推进理论、技术和方法创新，鼓励支持重大原创和颠覆性技术创新，催生一批高水平、原创性研究成果。发展社会服务力。在贡献和服务中体现价值，推动合作共建、多元投入的格局，大力推进政产学研用结合，强化科技成果转移转化及产业化。探索校城融合发展、校地联动发展的新模式，深度融入地方创新发展网络，为地方经济社会发展提供人才支撑，不断拓展和优化社会服务网络。涵育文化软实力。加快体制机制改革，优化学校、学部、学院三级评审机制，充分发挥优秀学者特别是德才兼备的年轻学者在学术治理中的重要作用。牢固树立一流意识、紧紧围绕一流目标、认真执行一流标准，让成就一流事业成为普遍追求和行动自觉。培育具有强大凝聚力的大学文化，营造积极团结、向上向善、干事创业的氛围，让大学成为吸引和留住一大批优秀人才建功立业的沃土，让敢干事、肯干事、能干事的人有更多的荣誉感和获得感。建设中国特色、世界一流大学不是等得来、喊得来的，而是脚踏实地拼出来、干出来的。对标一流，深化改革，坚持按章程办学，构建以一流质量标准为核心的制度规范体系，扎实推进学校综合改革，探索更具活力、更富效率的管理体制和运行机制，我们就一定能构建起具有中国特色的现代大学治理体系，进一步提升管理服务水平和工作效能。（作者系武汉大学校长）",
                 "育才造士，为国之本。党的干部是党和国家事业的中坚力量。习近平总书记深刻指出，“历史和现实都表明，一个政党、一个国家能不能不断培养出优秀领导人才，在很大程度上决定着这个政党、这个国家的兴衰存亡。”新修订的《党政领导干部选拔任用工作条例》，坚持以推进伟大事业为导向，将“事业为上、人岗相适、人事相宜”作为一条重要原则，为做好新时代选人用人工作、建设忠诚干净担当的高素质专业化干部队伍，进一步指明了正确方向。党的干部总是与党的事业紧紧连在一起，伟大事业需要高素质干部，干部要在事业发展中锻炼成长。党的十八大以来，党和国家事业之所以取得历史性成就、发生历史性变革，根本原因是有以习近平同志为核心的党中央坚强领导，有习近平新时代中国特色社会主义思想的科学指引，同时也与广大干部奋发有为、改革创新、干事创业、担当奉献密不可分。当前，中国特色社会主义进入新时代，站在新的历史起点上，我们党要肩负起新的历史使命，必须贯彻新时代党的组织路线，坚持从党和人民事业需要出发选干部、用干部，突出实践实干选贤能，坚持有为有位聚英才，真正做到事业发展需要什么样的人就用什么样的人，什么样的人最合适就选什么样的人。为官择人者治，为人择官者乱。新修订的《干部任用条例》，通篇贯穿事业导向、事业要求，在提出深入考察干部政治素质、道德品行、作风素养、廉政情况的同时，强调要突出工作实绩、履职尽责等方面的考察，大力选拔敢于负责、勇于担当、善于作为、实绩突出的优秀干部。贯彻落实条例，必须正确把握事业发展需要和干部成长进步的关系，知事识人、依事择人、精准选人，把善于统筹推进“五位一体”总体布局和协调推进“四个全面”战略布局，贯彻落实新发展理念、推进高质量发展、深化供给侧结构性改革、打好“三大攻坚战”的优秀干部及时发现出来、合理使用起来，进而示范引领更多干部勇于担当作为，心无旁骛干事业、坚定信心促发展。不拒众流，方为江海。五湖四海的事业，需要五湖四海的人来干。新修订的《干部任用条例》，鲜明提出要注意从企业、高等学校、科研院所等单位以及社会组织中发现选拔党政领导干部，对推动国有企事业单位、社会组织干部人才及时进入党政机关作出制度性安排，这是我们党选人用人成功经验的深刻总结和运用。贯彻落实条例，就要坚持干部工作一盘棋，进一步开阔视野、拓宽渠道，放眼各条战线、各个领域、各个行业、各个层级，充分盘活干部资源，广开进贤之路。要坚持公道正派、公正用人，选拔干部论能力、看水平、凭实绩，而不能搞平衡照顾、论资排辈、降格以求，更不能搞小圈子、任人唯亲，在少数人中选人。要突出政治过硬、本领高强，深入考察识别干部的专业能力、专业素养、专业精神。坚持立足当前、着眼长远，注重发现培养和选拔使用在改革发展稳定一线特别是在重大斗争中经过磨砺的优秀年轻干部，为党和国家事业发展源源不断地注入生机和活力。要坚持严管和厚爱结合、激励和约束并重，认真贯彻习近平总书记关于“三个区分开来”的重要要求，宽容干部在改革创新中的失误错误，保护干部干事创业的积极性。对那些符合有关规定给予容错的干部，要认真落实条例的要求，给予客观公正对待，为他们重整旗鼓、轻装上阵、贡献才智搭建平台。宏伟的事业，离不开高素质专业化的干部。各级党委（党组）及其组织人事部门，要以贯彻落实新修订的《干部任用条例》为契机，坚持党的原则第一、党的事业第一、人民利益第一，大力选拔党和人民需要的好干部，为推动中国特色社会主义伟大事业乘风破浪、不断前进提供坚强组织保证。《 人民日报 》"]
                
                for text in test_data:
                    print(self.model.sample_generate_encoder_decoder(text, add_eos=True, top_k=5))
                self.model.train()
                print("report loss is " + str(report_loss))
                report_loss = 0

            # 因为传入了target标签，因此会计算loss并且返回
            loss = self.model(token_ids,labels=labels_ids, decoder_input_ids=target_ids)[0]
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            report_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 8

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
