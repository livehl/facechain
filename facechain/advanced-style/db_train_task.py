import os
import shutil
import sys
import time
import traceback
import oss2

sys.path.append('../..')
from dbtool import sql_to_dict, update
from dbtool import setting as st
from setting import time_cache, uuid_str, loads
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

sys.path.append('../../facechain')


@time_cache(3600 * 24 * 30)
def get_oss() -> oss2.Bucket:
    """获取oss对象"""
    auth = oss2.Auth(st.ali_oss_id, st.ali_oss_secret)
    return oss2.Bucket(auth, st.ali_oss_endpoint, st.ali_oss_bucket, proxies=None)


def train_lora_fn(foundation_model_path=None, revision=None, output_img_dir=None, work_dir=None):
    os.system(
        f'PYTHONPATH=. accelerate launch train_text_to_image_lora.py --pretrained_model_name_or_path={foundation_model_path} '
        f'--revision={revision} --sub_path="film/film" '
        f'--output_dataset_name={output_img_dir} --caption_column="text" --resolution=512 '
        f'--random_flip --train_batch_size=1 --num_train_epochs=200 --checkpointing_steps=5000 '
        f'--learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 --seed=42 --output_dir={work_dir} '
        f'--lora_r=32 --lora_alpha=32 --lora_text_encoder_r=32 --lora_text_encoder_alpha=32')


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


def select_high_quality_face(input_img_dir):
    "选择高质量面部"
    quality_score_list = []
    abs_img_path_list = []
    face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa')

    for img_name in os.listdir(input_img_dir):
        if img_name.endswith('jsonl') or img_name.startswith('.ipynb'):
            continue
        abs_img_name = os.path.join(input_img_dir, img_name)
        face_quality_score = face_quality_func(abs_img_name)[OutputKeys.SCORES]
        if face_quality_score is None:
            quality_score_list.append(0)
        else:
            quality_score_list.append(face_quality_score[0])
        abs_img_path_list.append(abs_img_name)

    sort_idx = np.argsort(quality_score_list)[::-1]
    print('high quality face: ' + abs_img_path_list[sort_idx[0]])
    return abs_img_path_list[sort_idx[0]]


def main():
    while True:
        try:
            tasks = sql_to_dict("select * from facechain_lora where status=0 limit 1")
            oss: oss2.Bucket = get_oss()
            for task in tasks:
                print(task)
                update({"id": task.id, "status": 1}, "facechain_lora")
                work_path = f"lora_train/{task.uid}"
                images = []
                if task.images:
                    os.makedirs(work_path + "/raw_images/")
                for img in loads(task.images):
                    file_name = img.split("/")[-1]
                    local_img = work_path + "/raw_images/" + file_name
                    images.append(local_img)
                    oss.get_object_to_file(img, local_img)
                lora_path = tranin(work_path, images)
                update_file_name = st.face_lora_path + uuid_str() + "_weights.bin"
                oss.put_object_from_file(update_file_name, lora_path)
                # 获得最好的面部图片
                face_img=select_high_quality_face(work_path+"/training_data/personalizaition_lora_labeled")
                update_face_name = st.face_user_face_path + uuid_str() + "_fact.png"
                oss.put_object_from_file(update_face_name, face_img)
                update({"id": task.id, "lora": update_file_name,"face": update_face_name, "status": 2}, "facechain_lora")

            time.sleep(0.5)

        except KeyboardInterrupt:
            exit(0)
        except BaseException:
            traceback.print_exc()


if __name__ == "__main__":
    main()
