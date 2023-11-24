import sys
import torch
from PIL import Image, ImageDraw
import requests
from lavis.models import load_model_and_preprocess
from matplotlib import pyplot as plt
import re
import cv2
import numpy as np
from tqdm import tqdm
import os
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model


## get json content as dict
def get_json(file):
    import json
    with open(file, 'r', encoding='utf-8') as f:
        dicts = json.load(f)
    return dicts

def save_json(dicts, file, indent=2):
    import json
    info = json.dumps(dicts, indent=indent, ensure_ascii=False)
    with open(file, 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
        f.write(info)

def get_iou(box1,box2):
    in_h = min(box1[2],box2[2]) - max(box1[0],box2[0])
    in_w = min(box1[3],box2[3]) - max(box1[1],box2[1])
    
    inter = 0 if in_h < 0  or in_w < 0 else in_h * in_w
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    
    return inter / union
    

image_dire = './data/coco/images'


def get_predicts(json):
    templates = [
        "Question: Could you show me the {} in the image? Answer:",
        "Question: Could you present {} in the image? Answer:",
        "Question: Where is {} in the image? Answer:",
        "Question: Please show the region of {}? Answer:",
        "Question: Where is the {} in the image? Answer:",
        "Question: Could you identifying the {} locations? Answer:",
        "Question: Which region can I find {} in the image? Answer:",
        "Question: What are the coordinates of the {} in the image? Answer:",
        "Question: Could you locating the {} in this image? Answer:",
        "Question: Can you show me all the regions associated with {}? Answer:",
        "Question: Which area outlines the boundaries of the {}? Answer:",
        "Question: I would like to find {} in image. Can you give me its coordinates? Answer:",
        "Question: In the picture, I'd like you to locate {} and provide its coordinates. Answer:"
    ]
    
    no_finds = []
    for idx, item in tqdm(enumerate(json),total=len(json)):
        image =  Image.open(os.path.join(image_dire,item['image'])).convert('RGB')
        image = vis_processors["eval"](image).unsqueeze(0).to("cuda:0")
        print(image.device)
        flag = False
        for obj in item['object']:

            for template in templates:

                prompt = template.format(obj)


                predict = model.generate({ "image": image, "prompt": prompt})[0]
                coord = re.findall(r"\[(.*?)\]", predict)

                if not coord:
                    continue

                try:
                    coord = [int(float(i)) for i in coord[0].split(',')]
                except:
                    continue

                if len(coord) != 4:
                    continue

                iou = get_iou(coord,item['bbox'])

                if 'predict' in json[idx]:
                    json[idx]['predict'].append([coord, iou])
                else:
                    json[idx]['predict'] = [[coord,iou]]

                if iou >= 0.5:
                    flag = True
                    break
            
            if flag:
                break
            
        if not flag:
            no_finds.append(no_finds)
        
    return json


print('Initializing Chat')

def get_blip_model(device='cuda', dtype=torch.float16, use_multi_gpus=True):
    model, vis_processors, txt_processors = load_model_and_preprocess(
                    name='blip2_vicuna_instruct',
                    model_type='vicuna7b',
                    is_eval=True
                )

    ckpth = ""
    print(ckpth)
    print(model.load_checkpoint(ckpth)) 



    # model.to(dtype)
    if use_multi_gpus:
        # model.query_tokens = model.query_tokens.to('cuda:0')


        device_map = infer_auto_device_map(model, max_memory={0: "32GiB", 1: "32GiB",2: "32GiB"}, no_split_module_classes=['LlamaDecoderLayer', 'VisionTransformer'])
        print(device_map)
        print(device_map.keys())
        device_map['llm_model.lm_head'] = device_map['llm_proj'] = device_map['llm_model.model.embed_tokens'] = 0
        print(device_map)
        model = dispatch_model(model, device_map=device_map, offload_dir = 'vicuna-7b-all')
        torch.cuda.empty_cache()
    else:
        model.to('cuda:0')
    model.eval()
    return model,vis_processors,txt_processors

model, vis_processors, _ = get_blip_model()


testa = get_json('./data/refcoco/testA.json')
json = get_predicts(testa)
save_json(json,'./data/refcoco/testA-predict.json')

