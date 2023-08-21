import shutil
import sys
import time
import traceback

import cv2
import os
import oss2

sys.path.append('../..')
from dbtool import sql_to_dict, update, inserts, get
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


def inference_lora_fn(metadata, user_models, face, style_model: str, add_prompt: str, multiplier_style: float,
                      num_images: int):
    base_model = 'ly261666/cv_portrait_model'
    print("-------user_models: ", user_models)
    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False
    gen_portrait = GenPortrait(style_model, multiplier_style, add_prompt, use_main_model, use_face_swap,
                               use_post_process,
                               use_stylization)
    num_images = min(6, num_images)
    outputs = gen_portrait(metadata, face, num_images, base_model, user_models, 'film/film', 'v2.0')
    final_images = outputs["final"]
    outputs_RGB = []
    for out_tmp in final_images:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    outputs["final_rgb"] = outputs_RGB
    return outputs


def main():
    while True:
        try:
            tasks = sql_to_dict("select * from facechain_paint where status=0 limit 1")
            oss: oss2.Bucket = get_oss()
            for task in tasks:
                print(task)
                user_lora = get("facechain_lora", task.user_lora_id)
                print(user_lora)
                update({"id": task.id, "status": 1}, "facechain_paint")
                work_path = f"lora_inference/{user_lora.uid}"
                # 下载文件
                shutil.rmtree(work_path, ignore_errors=True)
                os.makedirs(work_path)
                lora_name = work_path + user_lora.lora.split("/")[-1]
                face_name = work_path + user_lora.face.split("/")[-1]
                oss.get_object_to_file(user_lora.lora, lora_name)
                oss.get_object_to_file(user_lora.face, face_name)
                result_data = inference_lora_fn(user_lora.metadata.split("\r\n"), lora_name, face_name, task.style_lora,
                                                task.add_prompt,
                                                task.multiplier_style, task.count)
                images = result_data["final_rgb"]
                paint_imgs = []
                for img in images:
                    file_name = st.face_img_path + "/" + uuid_str() + ".png"
                    paint_imgs.append({"uid": task.uid, "pid": task.pid, "path": file_name})
                    oss.put_object(file_name, img)
                update({"id": task.id, "status": 2}, "facechain_paint")
                inserts(paint_imgs, "facechain_paint_files")
            time.sleep(0.5)

        except KeyboardInterrupt:
            exit(0)
        except BaseException:
            traceback.print_exc()


if __name__ == "__main__":
    main()
