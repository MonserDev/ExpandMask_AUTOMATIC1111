import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

from modules import processing, images, shared, sd_samplers
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state, Options

from PIL import Image
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


class ExtensionTemplateScript(scripts.Script):
        # Extension title in menu UI

        def title(self):
                return "Extension Template"

        # Decide to show menu in txt2img or img2img
        # - in "txt2img" -> is_img2img is `False`
        # - in "img2img" -> is_img2img is `True`
        #
        # below code always show extension menu
        def show(self, is_img2img):
                return is_img2img
        
        

        # Setup menu ui detail
        def ui(self, is_img2img):

                def input_image_upload(input_image):
                

                        return input_image

                def apply_mask(sel_mask):

                        t = save_image(sel_mask)


                        return t
                
                def save_image(x):
                        # Assuming 'mask' is the key where the image data is stored in the dictionary
                        image_data = x.get('mask')

                        # Convert the image data to a PIL Image object
                        pil_image = Image.fromarray(np.uint8(image_data))

                        # Save the PIL Image to a file
                        pil_image.save(r"C:\Users\Monser\Desktop\test.png")

                        # Return the saved image path
                        return "C:\\Users\\Monser\\Desktop\\test.png"



                def expand_mask(input_image, sel_mask, expand_iteration=10):



                        new_sel_mask = sel_mask["mask"]


                        expand_iteration = int(np.clip(expand_iteration, 1, 100))

                        new_sel_mask = np.array(new_sel_mask, dtype=np.uint8)

                        new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

                        

                        return new_sel_mask
                
                with gr.Accordion('Extension Template', open=False):
                        with gr.Row():
                                Input_img = gr.Image()
                                Img_mask = gr.Image(label="Selected mask image", elem_id="ia_sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                                    show_label=False, interactive=True).style(height=480)
                                out = gr.Image()
                                Input_img.upload(input_image_upload,inputs=[Input_img],outputs=[Img_mask])

                        with gr.Column():


                                bt = gr.Button(value="Apply")
                                expand_bt = gr.Button(value="Expand")
                                expand_sli = gr.Slider(minimum=5,maximum=100,step=5)

                                bt.click(apply_mask, inputs=[Img_mask], outputs=[out])
                                expand_bt.click(expand_mask,inputs=[Input_img, Img_mask,expand_sli],outputs=[out])

                # TODO: add more UI components (cf. https://gradio.app/docs/#components)
                return [Img_mask, Input_img,out]

        # Extension main process
        # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
        # args is [StableDiffusionProcessing, UI1, UI2, ...]


        def run(self, Img_mask, Input_img,p,out):



                def convert_to_pil(rr):
                                # Assuming 'image' is the key where the image data is stored in the dictionary
    
                        pil_image = Image.fromarray(np.uint8(rr))
                                # Now you have the PIL Image object, and you can perform further operations
                        return pil_image
                
                img = cv2.imread("C:\\Users\\Monser\\Desktop\\test.png")
                img2 = convert_to_pil(img)

                print(type(img2))
                p.image_mask = img2
                p.latent_mask = None # fixes inpainting full resolution


                print("========mask_for_overlay===========\t" + str(type(p.image_mask)))
                print("========image_mask===========\t" + str(type(out)))
                print("=========init_image==========\t" + str(type(Input_img)))

                processed = processing.process_images(p)

                processed.images.append(p.image_mask)

                return processed
