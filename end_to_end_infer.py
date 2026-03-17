import os
import sys
import torch
import yaml
import argparse
import random
import copy
import re
import numpy as np
from PIL import Image
from transformers import BertTokenizer, BertModel

# --- Diffusers & PEFT Imports ---
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    DDPMScheduler, 
    AutoencoderKL,
    UNet2DConditionModel
)
from peft import PeftModel

# ==========================================
# 1. 环境与路径设置
# ==========================================
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

try:
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
except ImportError as e:
    print(f"[Error] 模块导入失败: {e}")
    sys.exit(1)

# ==========================================
# [核心] 200句全风格测试集 (无标点版)
# ==========================================
BATCH_POEMS = [
    # --- 原有 100 句 ---
    "明月松间照清泉石上流",
    "空山不见人但闻人语响",
    "独坐幽篁里弹琴复长啸",
    "人闲桂花落夜静春山空",
    "木末芙蓉花山中发红萼",
    "终南阴岭秀积雪浮云端",
    "野旷天低树江清月近人",
    "夜来风雨声花落知多少",
    "绿树村边合青山郭外斜",
    "湖光秋月两相和潭面无风镜未磨",
    "孤舟蓑笠翁独钓寒江雪",
    "青箬笠绿蓑衣斜风细雨不须归",
    "春潮带雨晚来急野渡无人舟自横",
    "千里莺啼绿映红水村山郭酒旗风",
    "停车坐爱枫林晚霜叶红于二月花",
    "借问酒家何处有牧童遥指杏花村",
    "小荷才露尖尖角早有蜻蜓立上头",
    "接天莲叶无穷碧映日荷花别样红",
    "儿童急走追黄蝶飞入菜花无处寻",
    "梅子金黄杏子肥麦花雪白菜花稀",
    "绿遍山原白满川子规声里雨如烟",
    "黄梅时节家家雨青草池塘处处蛙",
    "春色满园关不住一枝红杏出墙来",
    "横看成岭侧成峰远近高低各不同",
    "竹外桃花三两枝春江水暖鸭先知",
    "水光潋滟晴方好山色空蒙雨亦奇",
    "黑云翻墨未遮山白雨跳珠乱入船",
    "两个黄鹂鸣翠柳一行白鹭上青天",
    "迟日江山丽春风花草香",
    "黄四娘家花满蹊千朵万朵压枝低",
    "随风潜入夜润物细无声",
    "无边落木萧萧下不尽长江滚滚来",
    "岱宗夫如何齐鲁青未了",
    "暧暧远人村依依墟里烟",
    "采菊东篱下悠然见南山",
    "落英缤纷芳草鲜美",
    "杨花榆荚无才思惟解漫天作雪飞",
    "天街小雨润如酥草色遥看近却无",
    "几处早莺争暖树谁家新燕啄春泥",
    "一道残阳铺水中半江瑟瑟半江红",
    "大漠孤烟直长河落日圆",
    "秦时明月汉时关万里长征人未还",
    "青海长云暗雪山孤城遥望玉门关",
    "黄河远上白云间一片孤城万仞山",
    "白日依山尽黄河入海流",
    "葡萄美酒夜光杯欲饮琵琶马上催",
    "大漠穷秋塞草腓孤城落日斗兵稀",
    "千里黄云白日曛北风吹雁雪纷纷",
    "君不见走马川行雪海边平沙莽莽黄入天",
    "忽如一夜春风来千树万树梨花开",
    "故园东望路漫漫双袖龙钟泪不干",
    "今夜未知何处宿平沙万里绝人烟",
    "誓扫匈奴不顾身五千貂锦丧胡尘",
    "明月出天山苍茫云海间",
    "长安一片月万户捣衣声",
    "五月天山雪无花只有寒",
    "挽弓当挽强用箭当用长",
    "落日照大旗马鸣风萧萧",
    "黑云压城城欲摧甲光向日金鳞开",
    "夜阑卧听风吹雨铁马冰河入梦来",
    "渭城朝雨浥轻尘客舍青青柳色新",
    "孤帆远影碧空尽唯见长江天际流",
    "李白乘舟将欲行忽闻岸上踏歌声",
    "山随平野尽江入大荒流",
    "城阙辅三秦风烟望五津",
    "离离原上草一岁一枯荣",
    "寒蝉凄切对长亭晚骤雨初歇",
    "过春风十里尽荠麦青青",
    "旧时王谢堂前燕飞入寻常百姓家",
    "山围故国周遭在潮打空城寂寞回",
    "折戟沉沙铁未销自将磨洗认前朝",
    "烟笼寒水月笼沙夜泊秦淮近酒家",
    "潮落夜江斜月里两三星火是瓜洲",
    "月落乌啼霜满天江枫渔火对愁眠",
    "前不见古人后不见来者",
    "丞相祠堂何处寻锦官城外柏森森",
    "群山万壑赴荆门生长明妃尚有村",
    "江雨霏霏江草齐六朝如梦鸟空啼",
    "无情最是台城柳依旧烟笼十里堤",
    "楼船夜雪瓜洲渡铁马秋风大散关",
    "慈母手中线游子身上衣",
    "松下问童子言师采药去",
    "两句三年得一吟双泪流",
    "十年磨一剑霜刃未曾试",
    "江上往来人但爱鲈鱼美",
    "昨日入城市归来泪满巾",
    "陶尽门前土屋上无片瓦",
    "远看山有色近听水无声",
    "白毛浮绿水红掌拨清波",
    "过江千尺浪入竹万竿斜",
    "暮云收尽溢清寒银汉无声转玉盘",
    "墙角数枝梅凌寒独自开",
    "爆竹声中一岁除春风送暖入屠苏",
    "春风又绿江南岸明月何时照我还",
    "洛阳城里见秋风欲作家书意万重",
    "银烛秋光冷画屏轻罗小扇扑流萤",
    "荷尽已无擎雨盖菊残犹有傲霜枝",
    "儿童散学归来早忙趁东风放纸鸢",
    "牧童骑黄牛歌声振林樾",
    "头上红冠不用裁满身雪白走将来",
    
    # --- 题画诗库新增 100 句 ---
    "山色苍翠隐小屋溪流潺湲映古桥",
    "山光水色含清晖松影泉声带晚凉",
    "柳暗花明又一村山重水复疑无路",
    "山中何事松花落石上清泉带雨鸣",
    "月照松林影婆娑水映亭台夜色多",
    "松间石上唯闻水月下云中不见人",
    "竹影婆娑映石间清泉潺湲洗尘寰",
    "远山黛青水悠悠绿树掩映楼台幽",
    "雪山隐屋松间静流水绕村石上清",
    "菊影摇曳秋风起草间露滴映晨曦",
    "竹影婆娑风中舞石岩静默伴清幽",
    "山间松柏映茅庐远岫云烟绕翠微",
    "松间石上定僧禅云外人间看洞天",
    "云淡风轻山影远林深人静步声幽",
    "山高松密翠成堆涧静泉鸣声自回",
    "松间小屋临流水山下人家隔野烟",
    "岩壑盘纡入杳冥松萝交映众山青",
    "藤蔓垂果映僧影山岚轻笼远树青",
    "山雪未消春尚浅林烟初散日犹昏",
    "松枝盘曲引幽人竹影摇风伴寂心",
    "竹深荷净午风凉山高水满秋声早",
    "夏夜松间月影斜古寺钟声入梦来",
    "远山淡影连天际湖光树色映清波",
    "松下垂纶影入波远山云绕翠微多",
    "云破日出群山见龙飞天际万木春",
    "远山黛青村舍静流水绕田舟行轻",
    "塔影凌空接远山松声桥下水潺潺",
    "春晨雾绕山峦翠渔影轻摇水波微",
    "枯树临风诉岁寒山影入水映天蓝",
    "山间松影摇清风雪覆岩前映翠丛",
    "芦苇萧萧吹晚风水面平铺秋月明",
    "远山黛青云烟绕村舍依稀水岸旁",
    "枯树寒风摇落影红驹踏草映晨光",
    "山色空濛雨亦奇湖光潋滟晴方好",
    "远山淡影隐薄雾近水清流映雪岩",
    "松间石上定流水泉外风中忽野蝉",
    "飞流直下三千尺疑是银河落九天",
    "稻田金黄映翠林小河蜿蜒绕茅庐",
    "燕子飞时春事忙小桥流水野人家",
    "市桥官柳细依依江岸渔舟两两归",
    "楼阁参差树影斜水光浮动映晴霞",
    "石桥流水映花红小舟轻摇入画中",
    "竹影婆娑人独立云间仙鹤舞清风",
    "山寺隐于松柏间村水相映静无言",
    "秋风萧瑟稻花香山色苍茫鸟自翔",
    "绿树阴浓夏日长楼台倒影入池塘",
    "云海缭绕隐楼阁松影婆娑映翠微",
    "树影婆娑旗舞风市声鼎沸人如潮",
    "市井喧哗人语杂楼台灯火夜初长",
    "远山黛青水含烟秋树萧疏石径幽",
    "远山黛青松柏翠平野疏林茅舍幽",
    "竹影婆娑映石岩枯枝斜挂诉流年",
    "远山黛青近水明树影婆娑风自清",
    "秋风萧瑟芦花白雁影参差水自流",
    "远山黛青水如镜松桥横跨影婆娑",
    "牛食青草水映天树影摇风石上闲",
    "竹里梅花两三种鸟声清滑水泠泠",
    "鸟下绿芜秦苑夕蝉鸣黄叶汉宫秋",
    "红果垂枝映瀑流菊香竹影共清幽",
    "雪覆茅檐四五家竹间流水静鸣沙",
    "枯木寒枝映巨石牛车驴影伴人行",
    "秋风摇落黄叶飞白菊映日独芬菲",
    "骏影奔腾风中舞墨色飞动纸上生",
    "玉颜不及寒鸦色总为从前事萦惹",
    "墨竹幽幽生雅韵兰香袅袅绕清风",
    "水中游虾影婆娑岸畔清风送荷香",
    "竹影参差水岸斜石根苍翠藓痕加",
    "枫红石老秋意浓竹绿溪清心境融",
    "梅香引鸟枝头闹竹影摇风溪水清",
    "花间蝶舞影翩跹石上苔痕绿未干",
    "林间小溪映天黄岩畔松声伴晚凉",
    "竹里行厨洗玉盘花边立马簇金鞍",
    "花间蝶舞蜂飞忙草际蜻蜓点水凉",
    "骏马奔腾风卷尘长鬃飘逸日初升",
    "琴声绕梁醉宾客笑语盈室映华裳",
    "鹰踞岩巅松影斜松风拂羽静无哗",
    "翠竹摇风生清韵幽篁深处隐鸣泉",
    "雪覆山峰松影静林寒小屋隐幽深",
    "山楂红艳映岩青松风轻拂绕峰顶",
    "枯木参天叶尽凋竹影摇风声自萧",
    "岩间老树根如石竹里幽泉声似琴",
    "竹影参差风露清石根苍藓见秋晴",
    "草间鹌鹑隐幽踪石畔轻风拂翠丛",
    "藤蔓缠枝花烂漫溪流瀑布映红颜",
    "梅枝栖鸟赏清幽岩下细草映春柔",
    "鹤唳天高秋日明山静花飞晚节清",
    "枝头小雀舞清风岩下幽花映碧空",
    "秋晨茅屋隐松间雾绕山腰溪水潺",
    "塔影参差山色里松声断续水声中",
    "松枝盘曲映斜阳根须交错入土香",
    "芭蕉叶底清风起茅屋檐前白鹤飞",
    "荷影轻摇鸭戏水清风拂面叶生香",
    "芦花瑟瑟满汀洲鸟散渔舟自急流",
    "花间飞鸟忽双起枝上杨花吹又少",
    "玉兰花开春意浓绿叶扶疏映晴空",
    "古木阴中系短篷杖藜扶我过桥东",
    "小楼一夜听春雨深巷明朝卖杏花",
    "晨雾轻笼山间树屋檐滴翠竹影摇",
    "远浦帆归人未返江村树老鸟空啼",
    "飞鸟掠过淡雅天山丘静卧绿茵间"
]

# ==========================================
# 2. 辅助函数
# ==========================================

def calculate_total_iou(boxes_tensor):
    if boxes_tensor.size(0) < 2: return 0.0
    x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
    x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
    y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
    y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
    
    n = boxes_tensor.size(0)
    total_inter = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            xx1 = max(x1[i], x1[j]); yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j]); yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            total_inter += w * h
    return total_inter

def apply_random_symmetry(layout, device='cpu', attempt_prob=0.5):
    if not layout: return layout
    boxes_data = [list(item[1:5]) for item in layout] 
    boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32).to(device)
    initial_iou = calculate_total_iou(boxes_tensor)
    
    new_layout = copy.deepcopy(layout)
    current_boxes = boxes_tensor.clone()
    
    indices = list(range(len(layout)))
    random.shuffle(indices)
    
    for idx in indices:
        if random.random() > attempt_prob: continue
        original_item = new_layout[idx]
        original_box = current_boxes[idx].clone()
        
        new_cx = 1.0 - original_item[1]
        item_list = list(original_item)
        item_list[1] = new_cx
        
        if len(item_list) >= 9:
            item_list[5] = -item_list[5] # bias_x
            item_list[7] = -item_list[7] # rotation
        
        current_boxes[idx, 0] = new_cx
        new_iou = calculate_total_iou(current_boxes)
        
        if new_iou <= initial_iou + 1e-4: 
            new_layout[idx] = tuple(item_list)
            initial_iou = new_iou 
        else:
            current_boxes[idx] = original_box 
            
    return new_layout

def sanitize_filename(text):
    """提取汉字作为文件名"""
    safe_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    return safe_text[:10] if safe_text else "untitled_poem"

# ==========================================
# 3. 模型管线类
# ==========================================

class ShanshuiPipeline:
    def __init__(self, args):
        self.device = args.device
        self.args = args
        
        print("\n🚀 初始化全流程生成管线...")
        self.layout_model, self.tokenizer = self._load_layout_model()
        self.sd_pipe = self._load_sd_pipeline()
        self.mask_generator = InkWashMaskGenerator(width=args.width, height=args.height)
        
    def _load_layout_model(self):
        print(f"   [Stage 1] 加载布局模型: {self.args.layout_config}")
        with open(self.args.layout_config, "r") as f:
            config = yaml.safe_load(f)
        model_config = config['model']
        
        tokenizer = BertTokenizer.from_pretrained(model_config['bert_path'])
        
        model = Poem2LayoutGenerator(
            bert_path=model_config['bert_path'],
            num_classes=model_config['num_classes'],
            hidden_size=model_config['hidden_size'],
            bb_size=model_config['bb_size'],
            decoder_layers=model_config['decoder_layers'],
            decoder_heads=model_config['decoder_heads'],
            dropout=model_config['dropout'],
            latent_dim=model_config.get('latent_dim', 32)
        )
        
        checkpoint = torch.load(self.args.layout_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加入 strict=False 忽略由于版本不同导致的多余辅助权重层
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def _load_sd_pipeline(self):
        print(f"   [Stage 2] 加载 SD + ControlNet + LoRA (严格对齐训练验证模式)...")
        
        # 1. 像训练时一样单独加载所有组件
        tokenizer = BertTokenizer.from_pretrained(self.args.base_sd_path, subfolder="tokenizer")
# 【关键修复】统一指定为 float16
        text_encoder = BertModel.from_pretrained(self.args.base_sd_path, subfolder="text_encoder", torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(self.args.base_sd_path, subfolder="vae", torch_dtype=torch.float16)
        scheduler = DDPMScheduler.from_pretrained(self.args.base_sd_path, subfolder="scheduler")
        
        unet = UNet2DConditionModel.from_pretrained(
            self.args.base_sd_path, subfolder="unet", torch_dtype=torch.float16
        )
        
        lora_path = os.path.join(self.args.sd_checkpoint_dir, "unet_lora")
        try:
            unet = PeftModel.from_pretrained(unet, lora_path)
            # 🚫【核心修复】：绝对不要 merge_and_unload()，保留 PeftModel 状态防崩溃！
            print("   ✅ LoRA 挂载成功 (保留 PEFT 结构)")
        except Exception as e:
            print(f"   ❌ LoRA 挂载失败: {e}")
            sys.exit(1)
            
        controlnet_path = os.path.join(self.args.sd_checkpoint_dir, "controlnet_structure")
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        # 2. 组装 Pipeline
        pipe = StableDiffusionControlNetPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, controlnet=controlnet,
            scheduler=scheduler, safety_checker=None, feature_extractor=None
        ).to(self.device)
        
        # 🚫【核心修复】：这里不再覆盖为 UniPCMultistepScheduler，保留 DDPMScheduler
        
        if self.device == 'cuda':
            pipe.enable_model_cpu_offload()
        return pipe

    def decode_latents_to_image(self, latents):
        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        latents = 1 / scaling_factor * latents
        with torch.no_grad():
            image = self.sd_pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image[0])

    def generate(self, poem_text, seed=None, save_intermediates_dir=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        print(f"      正在推理: 【{poem_text[:15]}...】")
        
        # 1. Layout
        layout = greedy_decode_poem_layout(
            model=self.layout_model, 
            tokenizer=self.tokenizer, 
            poem=poem_text,
            max_elements=self.args.max_elements, 
            device=self.device
        )
        
        if not layout:
            print("      ⚠️ 无有效意象，跳过。")
            return None, None
            
        # [修复] 关闭可能破坏构图的随机镜像翻转
        # layout = apply_random_symmetry(layout, device=self.device, attempt_prob=0.6)

        # 2. Mask
        layout_list = [list(item) for item in layout]
        control_mask = self.mask_generator.convert_boxes_to_mask(layout_list)

        # 3. Diffusion
        # [修复] 严格对齐训练时的负面词
        n_prompt = "hard edges, sticker-like, flat color, cartoon, split screen, low quality, bad anatomy, 真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，杂乱，模糊，重影"
        
        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
            if save_intermediates_dir and (step % 5 == 0 or step == self.args.steps - 1):
                image = self.decode_latents_to_image(latents)
                step_str = str(step).zfill(3)
                save_path = os.path.join(save_intermediates_dir, f"step_{step_str}.png")
                image.save(save_path)

        callback = callback_fn if save_intermediates_dir else None
        callback_steps = 1

    # [修复] 严格对齐验证时的 0.85 强度和 0.7 早停晕染阈值，并加入 autocast 防止精度冲突
        with torch.autocast("cuda"):
            image = self.sd_pipe(
                prompt=poem_text,
                image=control_mask,
                negative_prompt=n_prompt,
                num_inference_steps=self.args.steps,
                guidance_scale=self.args.guidance,
                controlnet_conditioning_scale=0.85, 
                control_guidance_end=0.7, 
                width=self.args.width,
                height=self.args.height,
                generator=generator,
                callback=callback,
                callback_steps=callback_steps
            ).images[0]
        
        return image, control_mask

# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Poem2Painting Batch Inference")
    
    # 路径参数
    parser.add_argument('--layout_checkpoint', type=str, required=True, help="Stage 1 .pth")
    parser.add_argument('--sd_checkpoint_dir', type=str, required=True, help="Stage 2 Dir")
    parser.add_argument('--base_sd_path', type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument('--layout_config', type=str, default="configs/default.yaml")
    
    # 生成参数
    parser.add_argument('--output_dir', type=str, default="outputs/batch_100_test", help="结果保存目录")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--control_scale', type=float, default=1.0)
    parser.add_argument('--max_elements', type=int, default=30)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=None)
    
    # 开关
    parser.add_argument('--save_intermediates', action='store_true', help="保存中间过程")
    parser.add_argument('--single_poem', type=str, default=None, help="如果设置，只跑这一句")
    
    args = parser.parse_args()
    
    # 初始化
    pipeline = ShanshuiPipeline(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ===== [新增] 创建子文件夹 =====
    layouts_dir = os.path.join(args.output_dir, "layouts")
    paintings_dir = os.path.join(args.output_dir, "paintings")
    os.makedirs(layouts_dir, exist_ok=True)
    os.makedirs(paintings_dir, exist_ok=True)
    # ===============================

    # 确定要跑的列表
    if args.single_poem:
        tasks = [args.single_poem]
        print(f"\n🎯 单句测试模式: {args.single_poem}")
    else:
        tasks = BATCH_POEMS
        print(f"\n📚 批量测试模式: 共 {len(tasks)} 首诗")

    print(f"📂 结果输出总目录: {args.output_dir}")
    print(f"   - 布局文件夹: {layouts_dir}")
    print(f"   - 画作文件夹: {paintings_dir}\n")
    print("="*60)

    # 循环执行
    success_count = 0
    for i, poem in enumerate(tasks):
        safe_name = sanitize_filename(poem)
        prefix = f"{str(i+1).zfill(3)}_{safe_name}"
        
        print(f"[{i+1}/{len(tasks)}] 处理: {prefix}")
        
        # 中间过程目录 (暂时也保存在总目录下，或者可以按需移动)
        intermediates_dir = None
        if args.save_intermediates:
            intermediates_dir = os.path.join(args.output_dir, "steps", f"{prefix}_steps")
            os.makedirs(intermediates_dir, exist_ok=True)

        try:
            final_img, mask_img = pipeline.generate(
                poem, 
                seed=args.seed,
                save_intermediates_dir=intermediates_dir
            )
            
            if final_img:
                # ===== [新增] 将结果分别保存到对应的文件夹 =====
                save_path_img = os.path.join(paintings_dir, f"{prefix}_paint.png")
                save_path_mask = os.path.join(layouts_dir, f"{prefix}_mask.png")
                
                final_img.save(save_path_img)
                mask_img.save(save_path_mask)
                success_count += 1
                print(f"   ✅ 完成")
            else:
                print(f"   ⚠️ 跳过 (空布局)")
                
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)

    print(f"\n🎉 全部完成! 成功: {success_count}/{len(tasks)}")
    print(f"画作已保存至: {paintings_dir}")
    print(f"布局已保存至: {layouts_dir}")

if __name__ == "__main__":
    main()


    #python end_to_end_infer.py     --layout_checkpoint "/home/610-sty/layout2paint3/outputs/train_v11_bold_explore_rl/rl_best_reward.pth"     --sd_checkpoint_dir "/home/610-sty/l2p_plus/outputs/taiyi_shanshui_v19_breath/checkpoint-40000"     --base_sd_path "/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"     --layout_config "configs/default.yaml"     --output_dir "outputs/batch_100_v19_breath"     --max_elements 30     --control_scale 0.75     --guidance 7.5     --steps 50
