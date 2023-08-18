import enum
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import gradio as gr
import numpy as np
import torch
import oss2

sys.path.append('../..')
from dbtool import sql_to_dict, update, inserts
from dbtool import setting as st
from setting import time_cache, uuid_str
from inference import GenPortrait
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn

sys.path.append('../../facechain')


@time_cache(3600 * 24 * 30)
def get_oss() -> oss2.Bucket:
    """获取oss对象"""
    auth = oss2.Auth(st.ali_oss_id, st.ali_oss_secret)
    return oss2.Bucket(auth, st.ali_oss_endpoint, st.ali_oss_bucket, proxies=None)


def inference_lora_fn(work_path, user_models, style_model: str, add_prompt: str, multiplier_style: float,
                      num_images: int):
    base_model = 'ly261666/cv_portrait_model'
    print("-------user_models: ", user_models)
    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False
    output_model_name = 'personalizaition_lora'
    instance_data_dir = work_path + "/training_data/" + output_model_name
    gen_portrait = GenPortrait(style_model, multiplier_style, add_prompt, use_main_model, use_face_swap,
                               use_post_process,
                               use_stylization)
    num_images = min(6, num_images)
    outputs = gen_portrait(instance_data_dir, num_images, base_model, user_models, 'film/film', 'v2.0')
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    return outputs_RGB


def tranin(work_path: str, instance_images: list):
    output_model_name = 'personalizaition_lora'

    # mv user upload data to target dir
    instance_data_dir = f"{work_path}/training_data/{output_model_name}"
    print("--------instance_data_dir: ", instance_data_dir)
    work_dir = f"{work_path}/{output_model_name}"
    print("----------work_dir: ", work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(instance_data_dir, ignore_errors=True)
    prepare_dataset(instance_images, output_dataset_dir=instance_data_dir)
    data_process_fn(instance_data_dir, True)
    # train lora
    train_lora_fn(foundation_model_path='ly261666/cv_portrait_model',
                  revision='v2.0',
                  output_img_dir=instance_data_dir,
                  work_dir=work_dir)

    return work_dir + "/pytorch_lora_weights.bin"


def main():
    while True:
        try:
            tasks = sql_to_dict("select * from facechain_paint where status=0 limit 1")
            oss: oss2.Bucket = get_oss()
            for task in tasks:
                print(task)
                update({"id": task.id, "status": 1}, "facechain_paint")
                work_path = f"lora_inference/{task.uid}"
                images = inference_lora_fn(work_path, task.user_lora, task.style_lora, task.add_prompt,
                                           task.multiplier_style, task.count)
                paint_imgs = []
                for img in images:
                    file_name = st.face_img_path + "/" + uuid_str() + ".png"
                    paint_imgs.append({"uid": task.uid, "pid": task.pid, "path": file_name})
                    oss.put_object(file_name, img)
                update({"id": task.id, "status": 2}, "facechain_paint")
                inserts(paint_imgs, "facechain_paint_files")
            time.sleep(0.5)



        except BaseException:
            traceback.print_exc()


if __name__ == "__main__":
    main()
