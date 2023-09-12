# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from modelscope import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from torch import multiprocessing


from facechain.data_process.preprocessing import Blipv2
from facechain.merge_lora import merge_lora


def _data_process_fn_process(input_img_dir):
    Blipv2()(input_img_dir)


def data_process_fn(input_img_dir, use_data_process):
    ## TODO add face quality filter
    if use_data_process:
        ## TODO

        _process = multiprocessing.Process(target=_data_process_fn_process, args=(input_img_dir,))
        _process.start()
        _process.join()

    return os.path.join(str(input_img_dir) + '_labeled', "metadata.jsonl")


def txt2img(pipe, pos_prompt, neg_prompt, num_images=10):
    batch_size = 5
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe(prompt=pos_prompt, height=512, width=512, guidance_scale=7, negative_prompt=neg_prompt,
                            num_inference_steps=40, num_images_per_prompt=batch_size).images
        images_out.extend(images_style)
    return images_out


def img_pad(pil_file, fixed_height=512, fixed_width=512):
    w, h = pil_file.size

    if h / float(fixed_height) >= w / float(fixed_width):
        factor = h / float(fixed_height)
        new_w = int(w / factor)
        pil_file.thumbnail(size=(new_w, fixed_height))
        pad_w = int((fixed_width - new_w) / 2)
        pad_w1 = (fixed_width - new_w) - pad_w
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((0, 0), (pad_w, pad_w1), (0, 0)), 'constant')
    else:
        factor = w / float(fixed_width)
        new_h = int(h / factor)
        pil_file.thumbnail(size=(fixed_width, new_h))
        pad_h = fixed_height - new_h
        pad_h1 = 0
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((pad_h, pad_h1), (0, 0), (0, 0)), 'constant')

    output_file = Image.fromarray(array_file)
    return output_file


def txt2img_pose(pipe, pose_im, pos_prompt, neg_prompt, num_images=10):
    batch_size = 2
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe(prompt=pos_prompt, image=pose_im, height=512, width=512, guidance_scale=7,
                            negative_prompt=neg_prompt,
                            num_inference_steps=40, num_images_per_prompt=batch_size).images
        images_out.extend(images_style)
    return images_out


def txt2img_multi(pipe, images, pos_prompt, neg_prompt, num_images=10):
    batch_size = 2
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe(pos_prompt, images, height=512, width=512, guidance_scale=7, negative_prompt=neg_prompt,
                            controlnet_conditioning_scale=[1.0, 0.5],
                            num_inference_steps=40, num_images_per_prompt=batch_size).images
        images_out.extend(images_style)
    return images_out


def main_diffusion_inference(metadata, pos_prompt, neg_prompt, base_model_path, style_model_path, lora_model_path,
                             multiplier_style=0.25,
                             multiplier_human=0.85):
    pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')
    lora_style_path = style_model_path
    lora_human_path = lora_model_path
    pipe = merge_lora(pipe, lora_style_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_human_path, multiplier_human, from_safetensor=False)
    print(f'multiplier_style:{multiplier_style}, multiplier_human:{multiplier_human}')
    add_prompt_style = []
    tags_all = []
    cnt = 0
    cnts_trigger = np.zeros(6)
    for line in metadata:
        if line.strip():
            cnt += 1
            data = json.loads(line)['text'].split(', ')
            tags_all.extend(data)
            if data[1] == 'a boy':
                cnts_trigger[0] += 1
            elif data[1] == 'a girl':
                cnts_trigger[1] += 1
            elif data[1] == 'a handsome man':
                cnts_trigger[2] += 1
            elif data[1] == 'a beautiful woman':
                cnts_trigger[3] += 1
            elif data[1] == 'a mature man':
                cnts_trigger[4] += 1
            elif data[1] == 'a mature woman':
                cnts_trigger[5] += 1
            else:
                print('Error.')

    attr_idx = np.argmax(cnts_trigger)
    trigger_styles = ['a boy, children, ', 'a girl, children, ', 'a handsome man, ', 'a beautiful woman, ',
                      'a mature man, ', 'a mature woman, ']
    trigger_style = '<sks>, ' + trigger_styles[attr_idx]
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''
    # trigger_style = trigger_style + 'with <input_id> face, ' 
    # pos_prompt = 'Generate a standard ID photo of a chinese {}, solo, wearing high-class business/working suit, beautiful smooth face, with high-class/simple pure color background, looking straight into the camera with shoulders parallel to the frame, smile, high detail face, best quality, photorealistic'.format(gender)
    pipe = pipe.to("cuda")
    all_prompt = trigger_style + add_prompt_style + pos_prompt
    print(all_prompt)
    print(neg_prompt)
    images_style = txt2img(pipe, all_prompt, neg_prompt, num_images=10)

    return {"images": images_style, "prompt": all_prompt, "neg_prompt": neg_prompt}



def main_model_inference(pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human,
                         use_main_model, metadata,
                         base_model_path=None, lora_model_path=None):
    if use_main_model:
        return main_diffusion_inference(pos_prompt, neg_prompt, metadata, base_model_path, style_model_path,
                                        lora_model_path, multiplier_style=multiplier_style,
                                        multiplier_human=multiplier_human)


def face_swap_fn(use_face_swap, gen_results, template_face):
    if use_face_swap:
        ## TODO
        out_img_list = []
        image_face_fusion = pipeline('face_fusion_torch',
                                     model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.3')
        for img in gen_results:
            result = image_face_fusion(dict(template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            out_img_list.append(result)

        return out_img_list
    else:
        ret_results = []
        for img in gen_results:
            ret_results.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        return ret_results


def post_process_fn(use_post_process, swap_results_ori, selected_face, num_gen_images):
    if use_post_process:
        sim_list = []
        ## TODO
        face_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_ir_face-recognition-ood_rts', model_revision='v2.5')
        face_det_func = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd', model_revision='v1.1')
        swap_results = []
        for img in swap_results_ori:
            result_det = face_det_func(img)
            bboxes = result_det['boxes']
            if len(bboxes) == 1:
                bbox = bboxes[0]
                lenface = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                if 120 < lenface < 300:
                    swap_results.append(img)

        select_face_emb = face_recognition_func(selected_face)[OutputKeys.IMG_EMBEDDING][0]

        for img in swap_results:
            emb = face_recognition_func(img)[OutputKeys.IMG_EMBEDDING]
            if emb is None or select_face_emb is None:
                sim_list.append(0)
            else:
                sim = np.dot(emb, select_face_emb)
                sim_list.append(sim.item())
        sort_idx = np.argsort(sim_list)[::-1]

        return np.array(swap_results)[sort_idx[:min(int(num_gen_images), len(swap_results))]]
    else:
        return np.array(swap_results_ori)


class GenPortrait:
    def __init__(self, pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human,
                 use_main_model=True, use_face_swap=True,
                 use_post_process=True):
        self.use_main_model = use_main_model
        self.use_face_swap = use_face_swap
        self.use_post_process = use_post_process
        self.multiplier_style = multiplier_style
        self.multiplier_human = multiplier_human
        self.style_model_path = style_model_path
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt

    def __call__(self, metadata, face, num_gen_images=6, base_model_path=None,
                 lora_model_path=None, sub_path=None, revision=None):
        base_model_path = snapshot_download(base_model_path, revision=revision)
        if sub_path is not None and len(sub_path) > 0:
            base_model_path = os.path.join(base_model_path, sub_path)

        # main_model_inference PIL
        result_data = main_model_inference(self.pos_prompt, self.neg_prompt, self.style_model_path,
                                           self.multiplier_style,
                                           self.multiplier_human,
                                           self.use_main_model, metadata,
                                           lora_model_path=lora_model_path,
                                           base_model_path=base_model_path)
        images = result_data["images"]
        # select_high_quality_face PIL
        selected_face = Image.open(face)
        # face_swap cv2
        swap_results = face_swap_fn(self.use_face_swap, images, selected_face)
        # pose_process
        rank_results = post_process_fn(self.use_post_process, swap_results, selected_face,
                                       num_gen_images=num_gen_images)
        result_data["final"] = rank_results
        return result_data
