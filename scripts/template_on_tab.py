
import gc
import math
import os
import platform

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import re
import traceback

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionInpaintPipeline)

from modules import script_callbacks

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)

def test(x):
  print(type(x))

def input_image_upload(input_image, sel_mask):
    global sam_dict
    sam_dict["orig_image"] = input_image
    sam_dict["pad_mask"] = None


    return sam_dict["orig_image"]

def apply_mask(input_image, sel_mask):

    if input_image is not None and input_image.shape == sel_mask["mask"].shape:
        print("true")
        ret_image = cv2.addWeighted(input_image, 0.5, sel_mask["mask"], 0.5, 0)
    else:
        print("false")
        ret_image = sel_mask["mask"]

    

    return ret_image



def expand_mask(input_image, sel_mask, expand_iteration=10):



    new_sel_mask = sel_mask["mask"]


    expand_iteration = int(np.clip(expand_iteration, 1, 100))

    new_sel_mask = np.array(new_sel_mask, dtype=np.uint8)

    new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

    

    return new_sel_mask


def on_ui_tabs():
    with gr.Blocks() as demo:
        with gr.Row():
            Input_img = gr.Image()
            Img_mask = gr.Image(label="Selected mask image", elem_id="ia_sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                                    show_label=False, interactive=True).style(height=480)
            out = gr.Image(type="numpy")
            Input_img.upload(input_image_upload,inputs=[Input_img],outputs=[Img_mask])

        with gr.Column():


            bt = gr.Button(value="Apply")
            expand_bt = gr.Button(value="Expand")
            expand_sli = gr.Slider(minimum=5,maximum=100,step=5)

            bt.click(apply_mask, inputs=[Input_img, Img_mask], outputs=[out])
            expand_bt.click(expand_mask,inputs=[Input_img, Img_mask,expand_sli],outputs=[out])


        return [(demo, "Extension Template", "extension_template_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
