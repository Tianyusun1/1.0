import os
import sys
import torch
import yaml
import argparse
import random
import copy
import numpy as np
from PIL import Image
from transformers import BertTokenizer

# --- Diffusers & PEFT Imports ---
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler,
    UNet2DConditionModel
)
from peft import PeftModel

# ==========================================
# 1. 环境与路径设置
# ==========================================
# 获取当前脚本路径，确保能导入项目模块
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

try:
    # Stage 1 Imports
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    
    # Stage 2 Imports
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
except ImportError as e:
    print(f"[Error] 模块导入失败: {e}")
    print("请确保脚本位于项目根目录，并且 models/, inference/, stage2_generation/ 文件夹存在。")
    sys.exit(1)

# ==========================================
# 2. 辅助函数 (Stage 1)
# ==========================================

def calculate_total_iou(boxes_tensor):
    """计算所有框的总重叠面积"""
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
    """尝试对布局进行水平翻转（增加构图多样性）"""
    if not layout: return layout
    # 提取 Box 数据用于计算 IoU
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
        
        # 翻转逻辑: cx' = 1 - cx
        new_cx = 1.0 - original_item[1]
        
        # 翻转态势: Bias_X (Rotation Bias) 取反
        item_list = list(original_item)
        item_list[1] = new_cx
        
        # 假设 layout item 格式: [cls, cx, cy, w, h, bx, by, rot, flow]
        if len(item_list) >= 9:
            item_list[5] = -item_list[5] # bias_x 取反
            item_list[7] = -item_list[7] # rotation 镜像
        
        current_boxes[idx, 0] = new_cx
        new_iou = calculate_total_iou(current_boxes)
        
        # 只有当翻转不导致严重的重叠增加时才接受
        if new_iou <= initial_iou + 1e-4: 
            new_layout[idx] = tuple(item_list)
            initial_iou = new_iou 
        else:
            current_boxes[idx] = original_box # 撤销
            
    return new_layout

# ==========================================
# 3. 模型加载类
# ==========================================

class ShanshuiPipeline:
    def __init__(self, args):
        self.device = args.device
        self.args = args
        
        print("\n🚀 初始化全流程生成管线...")
        
        # --- 加载 Stage 1: 布局生成模型 ---
        self.layout_model, self.tokenizer = self._load_layout_model()
        
        # --- 加载 Stage 2: 绘画生成模型 (PEFT + ControlNet) ---
        self.sd_pipe = self._load_sd_pipeline()
        
        # --- 工具: 墨韵掩码生成器 ---
        self.mask_generator = InkWashMaskGenerator(width=args.width, height=args.height)
        
    def _load_layout_model(self):
        print(f"   [Stage 1] 加载布局模型配置: {self.args.layout_config}")
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
        
        print(f"   [Stage 1] 加载权重: {self.args.layout_checkpoint}")
        checkpoint = torch.load(self.args.layout_checkpoint, map_location=self.device)
        
        # 处理 state_dict 键名 (移除 module. 前缀)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def _load_sd_pipeline(self):
        print(f"   [Stage 2] 加载基础模型: {self.args.base_sd_path}")
        
        # 1. 加载 Base UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.args.base_sd_path, subfolder="unet", torch_dtype=torch.float16
        )
        
        # 2. 挂载 PEFT LoRA (核心步骤)
        lora_path = os.path.join(self.args.sd_checkpoint_dir, "unet_lora")
        print(f"   [Stage 2] 挂载 LoRA 权重: {lora_path}")
        try:
            unet = PeftModel.from_pretrained(unet, lora_path)
            unet = unet.merge_and_unload() # 物理融合
            print("   ✅ LoRA 融合成功")
        except Exception as e:
            print(f"   ❌ LoRA 挂载失败: {e}")
            sys.exit(1)
            
        # 3. 加载 ControlNet
        controlnet_path = os.path.join(self.args.sd_checkpoint_dir, "controlnet_structure")
        print(f"   [Stage 2] 加载 ControlNet: {controlnet_path}")
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        # 4. 组装 Pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.args.base_sd_path,
            unet=unet, # 注入了 LoRA 的 UNet
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # 启用显存优化
        if self.device == 'cuda':
            pipe.enable_model_cpu_offload()
            
        return pipe

    # [NEW] 将 Latents 解码为可见图片的辅助函数
    def decode_latents_to_image(self, latents):
        # SD 默认缩放因子
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

        print(f"\n🎨 正在处理诗句: 【{poem_text}】")
        
        # --- Step 1: Layout Generation ---
        print("   1. 生成布局 (Layout)...")
        layout = greedy_decode_poem_layout(
            model=self.layout_model, 
            tokenizer=self.tokenizer, 
            poem=poem_text,
            max_elements=self.args.max_elements, 
            device=self.device
        )
        
        if not layout:
            print("   ⚠️ 警告: 未生成有效布局，跳过。")
            return None, None
            
        # 随机对称增强
        layout = apply_random_symmetry(layout, device=self.device, attempt_prob=0.6)
        print(f"      生成了 {len(layout)} 个意象元素。")

        # --- Step 2: Mask Generation ---
        print("   2. 生成墨韵掩码 (Ink Mask)...")
        layout_list = [list(item) for item in layout]
        control_mask = self.mask_generator.convert_boxes_to_mask(layout_list)

        # --- Step 3: Image Diffusion ---
        print("   3. 扩散生成画作 (Diffusion)...")
        n_prompt = "真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，杂乱，模糊，重影"
        
        # --- [NEW] 定义回调函数保存中间过程 ---
        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor):
            # 每 5 步保存一次，或者是最后一步
            if save_intermediates_dir and (step % 5 == 0 or step == self.args.steps - 1):
                image = self.decode_latents_to_image(latents)
                step_str = str(step).zfill(3)
                save_path = os.path.join(save_intermediates_dir, f"step_{step_str}.png")
                image.save(save_path)

        callback = callback_fn if save_intermediates_dir else None
        # 如果设置了回调，步长设为1以确保能捕捉
        callback_steps = 1

        image = self.sd_pipe(
            prompt=poem_text,
            image=control_mask,
            negative_prompt=n_prompt,
            num_inference_steps=self.args.steps,
            guidance_scale=self.args.guidance,
            controlnet_conditioning_scale=self.args.control_scale,
            width=self.args.width,
            height=self.args.height,
            generator=generator,
            callback=callback,          # 注入回调
            callback_steps=callback_steps # 设置频率
        ).images[0]
        
        return image, control_mask

# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Poem2Painting End-to-End Inference")
    
    # 路径参数
    parser.add_argument('--layout_checkpoint', type=str, required=True, help="Stage 1 Poem2Layout .pth 文件路径")
    parser.add_argument('--sd_checkpoint_dir', type=str, required=True, help="Stage 2 Checkpoint 目录 (包含 unet_lora 和 controlnet_structure)")
    parser.add_argument('--base_sd_path', type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", help="太乙 SD 底座模型路径")
    parser.add_argument('--layout_config', type=str, default="configs/default.yaml", help="布局模型配置文件")
    
    # 生成参数
    parser.add_argument('--poem', type=str, default="两只黄鹂鸣翠柳，一行白鹭上青天。", help="输入诗句")
    parser.add_argument('--output_dir', type=str, default="outputs/final_results", help="结果保存目录")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--steps', type=int, default=30)
    
    # [FIXED] 修正参数类型错误 type.float -> type=float
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--control_scale', type=float, default=0.8)
    
    parser.add_argument('--max_elements', type=int, default=30)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=None)
    
    # [NEW] 是否保存中间过程
    parser.add_argument('--save_intermediates', action='store_true', help="开启后，将在输出目录创建子文件夹保存扩散过程图")
    
    args = parser.parse_args()
    
    # 初始化管线
    pipeline = ShanshuiPipeline(args)
    
    # 执行生成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 简单的文件名清理
    safe_name = "".join([c for c in args.poem if c.isalnum()])[:10]
    if not safe_name: safe_name = "demo_result"

    # 如果需要保存中间过程，创建子目录
    intermediates_dir = None
    if args.save_intermediates:
        intermediates_dir = os.path.join(args.output_dir, f"{safe_name}_steps")
        os.makedirs(intermediates_dir, exist_ok=True)
        print(f"   📂 中间过程将保存在: {intermediates_dir}")
    
    # 执行生成
    final_img, mask_img = pipeline.generate(
        args.poem, 
        seed=args.seed,
        save_intermediates_dir=intermediates_dir
    )
    
    if final_img:
        # 保存
        save_path_img = os.path.join(args.output_dir, f"{safe_name}_paint.png")
        save_path_mask = os.path.join(args.output_dir, f"{safe_name}_mask.png")
        
        final_img.save(save_path_img)
        mask_img.save(save_path_mask)
        
        print(f"✅ 结果已保存:")
        print(f"   画作: {save_path_img}")
        print(f"   掩码: {save_path_mask}")
        if intermediates_dir:
            print(f"   过程: {intermediates_dir}/")

if __name__ == "__main__":
    main()