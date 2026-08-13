import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-230M-8k").to(device)
model.eval()

# 2. 加载tokenizer和image_processor
tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, 
                          model.cfg.lm_chat_template)
image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, 
                                      model.cfg.resize_to_max_side_len)

# 3. 加载频谱图（.npy → RGB）
npy_path = r"D:\AfterXTU\Project_From_nanoVLM\36_LTE_1/processed/spectrograms/IQ_2024-04-20T22-06-22_790713246.npy"
spec = np.load(npy_path)
plt.imsave("temp_spec.jpg", spec, cmap='jet')
img = Image.open("temp_spec.jpg").convert('RGB')

# 4. 预处理图像（完全复制官方generate.py的写法）
processed_image, splitted_image_ratio = image_processor(img)
if not hasattr(tokenizer, "global_image_token") and splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
    processed_image = processed_image[1:]

# 5. 构造prompt（使用官方的get_image_string）
image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)
prompt = image_string + "What do you see in this spectrum?"

# 6. 生成输出（使用官方的apply_chat_template）
messages = [{"role": "user", "content": prompt}]
encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = encoded_prompt.input_ids.to(device)
img_t = processed_image.to(device)

output_ids = model.generate(input_ids, img_t, max_new_tokens=50)
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print("模型回答:", response)
##没事我感觉我对于python了解更深了，帮我总结一下我从设置解释器（我采用的是你教我的找python.exe的方法），到pip install 完，再到这段代码成功运行的所有经验做一个总结