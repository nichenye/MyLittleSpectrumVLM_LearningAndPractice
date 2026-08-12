import argparse
import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    #代码里锁死了“随机种子”
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


def parse_args():
    # 创建一个“参数解析器”，描述这个脚本是“从图片生成文本”
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    
    # --checkpoint: 本地模型路径（如果有的话）
    # 默认是 None，表示从 HuggingFace 下载
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    
    # --hf_model: HuggingFace 上的模型仓库名
    # 默认是 lusxvr/nanoVLM-230M-8k（你刚才下载的那个 912MB 的模型）
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    
    # --image: 输入图片路径
    # 默认是 assets/image.png（那只猫）
    parser.add_argument("--image", type=str, default="assets/image.png",
                        help="Path to input image")
    
    # --prompt: 你对图片提的问题
    # 默认是 "What is this?"（你改成 "Describe this panda" 就能问熊猫了）
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    
    # --generations: 生成几条不同的回答
    # 默认是 5（所以你刚才看到了 5 条描述）
    parser.add_argument("--generations", type=int, default=1,
                        help="Num. of outputs to generate")
    
    # --max_new_tokens: 每条回答最多生成多少个词（token）
    # 默认 300，所以每段描述都不会太长
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    
    # --measure_vram: 是一个“开关”，如果加上就会显示显存占用
    # 默认是 False（不显示），加上 --measure_vram 才会显示
    parser.add_argument("--measure_vram", action="store_true",
                        help="Measure and display VRAM usage during model loading and generation")
    
    # 把所有参数解析成一个对象，返回给 main() 使用
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize()
        model_vram_bytes = torch.cuda.memory_allocated(device)
        model_vram_mb = model_vram_bytes / (1024 ** 2)
        print(f"VRAM used after loading model: {model_vram_mb:.2f} MB")

    # Get tokenizer and image processor from model config if not provided
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    resize_to_max_side_len = False
    if hasattr(model.cfg, "resize_to_max_side_len"):
        resize_to_max_side_len = model.cfg.resize_to_max_side_len
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, resize_to_max_side_len)

    img = Image.open(args.image).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(img)
    if not hasattr(tokenizer, "global_image_token") and splitted_image_ratio[0]*splitted_image_ratio[1] == len(processed_image) - 1:
        # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + args.prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # 关键修复：提取 input_ids，确保是 torch.LongTensor
    tokens = encoded_prompt.input_ids.to(device)
    img_t = processed_image.to(device)

    print("\nInput:\n ", args.prompt, "\n\nOutput:")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        
        if args.measure_vram and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_mb = peak_vram_bytes / (1024 ** 2)
            current_vram_bytes = torch.cuda.memory_allocated(device)
            current_vram_mb = current_vram_bytes / (1024 ** 2)
            print(f"  >> Generation {i+1}: {out}")
            print(f"     VRAM - Peak: {peak_vram_mb:.2f} MB, Current: {current_vram_mb:.2f} MB")
        else:
            print(f"  >> Generation {i+1}: {out}")

if __name__ == "__main__":
    main()