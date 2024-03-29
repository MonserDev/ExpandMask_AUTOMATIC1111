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


class Script(scripts.Script):
        # Extension title in menu UI

        def title(self):
                return "ExpandMask"

        # Decide to show menu in txt2img or img2img
        # - in "txt2img" -> is_img2img is `False`
        # - in "img2img" -> is_img2img is `True`
        #
        # below code always show extension menu
        def show(self, is_img2img):
                return is_img2img
        
        

        # Setup menu ui detail
        def ui(self, is_img2img):

                expand_mask_blur = gr.Slider(label="Maskblur",minimum=0,maximum=100,step=2,value=10)
                expand_sli = gr.Slider(label="Expand",minimum=5,maximum=100,step=5,value=10)

                # TODO: add more UI components (cf. https://gradio.app/docs/#components)
                return [expand_sli,expand_mask_blur]

        # Extension main process
        # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
        # args is [StableDiffusionProcessing, UI1, UI2, ...]


        def run(self,p,expand_sli,expand_mask_blur):

                # def save_image_noget(x):
                #         # Assuming 'mask' is the key where the image data is stored in the dictionary
                #         image_data = x

                #         # Convert the image data to a PIL Image object
                #         pil_image = Image.fromarray(np.uint8(image_data))

                #         # Save the PIL Image to a file
                #         pil_image.save('D:\\Monser-sdee-ui\\Monser-sdee-ui\\temp.png')

                #         # Return the saved image path
                #         return pil_image

                def expand_mask(sel_mask, expand_iteration=10):

                        if not type(sel_mask) != "PIL.Image.Image":
                                new_sel_mask = sel_mask["mask"]
                        else:
                                new_sel_mask = sel_mask


                        expand_iteration = int(np.clip(expand_iteration, 1, 100))

                        new_sel_mask = np.array(new_sel_mask, dtype=np.uint8)

                        new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

                        new_sel_mask = Image.fromarray(np.uint8(new_sel_mask))

                        # save_image_noget(new_sel_mask)                      

                        return new_sel_mask

                # def convert_to_pil(Image):
                #                 # Assuming 'image' is the key where the image data is stored in the dictionary
                #         image_data = Image.get('mask')
                #         pil_image = Image.fromarray(np.uint8(image_data))
                #         save_image_noget(pil_image)
                #                 # Now you have the PIL Image object, and you can perform further operations
                #         return pil_image
                
                p.image_mask = expand_mask(p.image_mask,expand_sli)
                img_mask = p.image_mask
                p.mask_blur = expand_mask_blur
                proc = process_images(p)

                proc.images.append(img_mask)


                #p.init_images[0]


                # TODO: add image edit process via Processed object proc
                return proc
