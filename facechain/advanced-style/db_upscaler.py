import traceback

from PIL import Image
import torch
import shutil
from dbtool import sql_to_dict, update, inserts, get
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline
import oss2
import os
import time

from dbtool import setting as st
from setting import time_cache, uuid_str, dumps


def up_4x_pic(prompt: str, img: Image):
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(
        "system_lora/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
    upscaler.to("cuda")
    return upscaler(prompt=prompt, image=img).images[0]


def up_2x_pic(prompt: str, img: Image):
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "system_lora/sd-x2-latent-upscaler", torch_dtype=torch.float16)
    upscaler.to("cuda")
    return upscaler(prompt=prompt, image=img).images[0]


@time_cache(3600 * 24 * 30)
def get_oss() -> oss2.Bucket:
    """获取oss对象"""
    auth = oss2.Auth(st.ali_oss_id, st.ali_oss_secret)
    return oss2.Bucket(auth, st.ali_oss_endpoint, st.ali_oss_bucket, proxies=None)


def main():
    while True:
        try:
            tasks = sql_to_dict("select * from facechain_upscale where status=0 limit 1")
            oss: oss2.Bucket = get_oss()
            for task in tasks:
                print(task)
                update({"id": task.id, "status": 1}, "facechain_upscale")
                work_path = f"lora_inference/upscale"
                # 下载文件
                shutil.rmtree(work_path, ignore_errors=True)
                os.makedirs(work_path)
                face_path = work_path + task.path.split("/")[-1]
                oss.get_object_to_file(task.path, face_path)
                image = up_2x_pic(task.prompt, Image.open(face_path))
                file_name = task.path.split("/")[:-2] + task.path.split("/")[-1].split(".")[0] + "_up_2x.png"
                oss.put_object(file_name, image)
                update({"id": task.id, "status": 2, "upscale_path": file_name}, "facechain_upscale")
            time.sleep(0.5)

        except KeyboardInterrupt:
            exit(0)
        except BaseException:
            traceback.print_exc()


if __name__ == "__main__":
    main()
