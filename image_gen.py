# @markdown #**Preparation**

import cv2
import torch
import time
import os
import uvicorn
import tempfile

from utils.inference.image_processing import crop_face, get_final_image, show_images
from utils.inference.video_processing import (
    read_video,
    get_target,
    get_final_video,
    add_audio_from_another_video,
    face_enhancement,
)
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from coordinate_reg.image_infer import Handler
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions
import warnings

warnings.filterwarnings("ignore")
import fastapi
import numpy as np
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import File, UploadFile
import base64
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def get_app():
    app = fastapi.FastAPI()
    return app


app = get_app()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class ImageGen:
    def __init__(self):
        self.load_models()

    # init models for server
    def load_models(self):
        self.app = Face_detect_crop(name="antelope", root="./insightface_func/models")
        self.app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

        # main model for generation
        self.G = AEI_Net(backbone="unet", num_blocks=2, c_id=512)
        self.G.eval()
        self.G.load_state_dict(
            torch.load("./weights/G_unet_2blocks.pth", map_location=torch.device("cpu"))
        )

        # arcface model to get face embedding
        self.netArc = iresnet100(fp16=False)
        self.netArc.load_state_dict(
            torch.load("./arcface_model/backbone.pth", map_location=torch.device("cpu"))
        )
        # put to cpu
        self.netArc.eval()

        # # model to get face landmarks
        self.handler = Handler(
            "./coordinate_reg/model/2d106det", 0, ctx_id=0, det_size=640, cpu=True
        )


    def generate_image(self, source, target):
        crop_size = 224  # don't change this

        # check, if we can detect face on the source image
        try:
            source = crop_face(source, self.app, crop_size)[0]
            source = [source[:, :, ::-1]]

        except Exception as e:
            logger.error(e)
            print("Bad source images")

        full_frames = [target]

        target = get_target(full_frames, self.app, crop_size)

        batch_size = 100

        START_TIME = time.time()

        final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
            full_frames,
            source,
            target,
            self.netArc,
            self.G,
            self.app,
            set_target=False,
            crop_size=crop_size,
            BS=batch_size,
        )

        result = get_final_image(
            final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, self.handler
        )

        return result


# init models for server
image_gen = ImageGen()


# create api to generate image
# recieve image files and return generated image
@app.post("/generate_image/")
async def generate_image(source: UploadFile = File(...), target: UploadFile = File(...)):
    try:
        source = await source.read()
        target = await target.read()
        source = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
        target = cv2.imdecode(np.frombuffer(target, np.uint8), cv2.IMREAD_COLOR)
        result = image_gen.generate_image(source, target)
        _, im_arr = cv2.imencode(".jpg", result[:, :, ::-1])  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        
        # save in temp file
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            filename = tmp.name
            cv2.imwrite(tmp.name, result)

        return FileResponse(filename)
    
    except Exception as e:
        print(e)
        return {"result": "Error in captioning image"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8500)


if __name__ == "__main__":
    main()
