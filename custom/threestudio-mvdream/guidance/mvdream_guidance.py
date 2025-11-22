import sys
from dataclasses import dataclass, field
from typing import List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import numpy as np
import threestudio
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image # <<< Import save_image
from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
import math
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import torchvision.transforms as transforms
import inspect

@threestudio.register("mvdream-multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5
        qwen_loss: float = -1.0
        qwen_prob_loss: bool = False
        prompt_type: int = 1
        sds_prompt_not_none: bool = True
        vlm_prompt: str = "An orange and a red apple are on a plate, a green apple is beside the plate, all on a table."
        
        save_preview_every: int = 10
        save_preview_dir: str = "threestudio-outputs/previews"


    cfg: Config


    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")
        print('MultiviewDiffusionGuidance Config', self.cfg)
        
        self.call_counter = 0
        # <<< Create the preview save directory >>>
        os.makedirs(self.cfg.save_preview_dir, exist_ok=True)
        print(f"Preview images will be saved to {self.cfg.save_preview_dir} every {self.cfg.save_preview_every} calls.")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path).to(
            self.device
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        if self.cfg.qwen_loss > 0.0:
            ckpt_path = "./models/Qwen2.5-VL-7B-Instruct"
            self.processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
            self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            self.vlm_model.eval().requires_grad_(False).to(self.device)
            print('Loading Qwen Model', ckpt_path, self.cfg.qwen_loss)
            try:
                if hasattr(self.processor, 'image_processor') and self.processor.image_processor is not None:
                    img_processor_obj = self.processor.image_processor
                    img_processor_class = img_processor_obj.__class__
                    img_processor_name = img_processor_class.__name__

                    try:
                        img_processor_file_path = inspect.getfile(img_processor_class)
                    except TypeError:
                        img_processor_file_path = "N/A"

                    print("\n--- Image Processor Details ---")
                    print(f"  Actual Class Name: {img_processor_name}")
                    print(f"  Defined in File: {img_processor_file_path}")
                    print("-----------------------------\n")

                else:
                    print("\n--- Image Processor Details ---")
                    print("  'self.processor' does not have an 'image_processor' attribute or it is None.")
                    print("-----------------------------\n")

            except Exception as e:
                print(f"\n--- Error inspecting image_processor: {e} ---")

        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def smart_resize(self, height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056):
        if height < factor or width < factor:
            factor = min(height, width, factor) if min(height, width, factor) > 0 else 1
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar == 0: h_bar = factor
        if w_bar == 0: w_bar = factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt(max(1, height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
            if h_bar == 0: h_bar = factor
            if w_bar == 0: w_bar = factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / max(height * width, 1))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
            if h_bar == 0: h_bar = factor
            if w_bar == 0: w_bar = factor
        h_bar = int(max(factor, h_bar))
        w_bar = int(max(factor, w_bar))
        h_bar = round(h_bar / factor) * factor
        w_bar = round(w_bar / factor) * factor
        return int(h_bar), int(w_bar)

    def prepare_initial_image_tensors(self, image_tensor, processor):
        img_processor = processor.image_processor
        patch_size = img_processor.patch_size
        merge_size = img_processor.merge_size
        h_orig, w_orig = image_tensor.shape[-2], image_tensor.shape[-1]
        target_h, target_w = self.smart_resize(
            h_orig, w_orig, factor=patch_size * merge_size,
            min_pixels=img_processor.min_pixels, max_pixels=img_processor.max_pixels
        )

        img_transforms = transforms.Compose([
            transforms.Resize([target_h, target_w], interpolation=img_processor.resample, antialias=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean=img_processor.image_mean, std=img_processor.image_std)
        ])
        image1_tensor_norm = img_transforms(image_tensor)
        return image1_tensor_norm

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        # <<< Increment call counter >>>
        self.call_counter += 1
        
        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False) * 2 - 1)
            else:
                # interp to 256x256 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                # print('pred_rgb', pred_rgb.shape, pred_rgb.min(), pred_rgb.max(), pred_rgb.mean()) # torch.Size([8, 3, 256, 256]), range[0, 1]
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = 1 - self.alphas_cumprod[t]
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        reward_loss = torch.tensor(0.0, device=latents.device)  # Initialize reward loss
        if self.cfg.qwen_loss > 0.0 and self.call_counter <= 10000:
            # img = pred_rgb[0].unsqueeze(0).clamp(0, 1)
            # print('img', img.shape) # torch.Size([1, 3, 512, 512])
            img = pred_rgb.clamp(0, 1)
            if self.cfg.sds_prompt_not_none:
                contexts = prompt_utils.prompt
            else:
                contexts = self.cfg.vlm_prompt

            # print('contexts', contexts)

            if self.cfg.prompt_type == 1:
                self.prompt_text = (
                    "Please carefully evaluate the following images showing multiple views rendered from a single 3D target.\n"
                    "Determine whether the underlying 3D object, as represented by all these views collectively, simultaneously meets all these conditions:\n"
                    f"1. Content Match: The object primarily depicts {contexts}.\n"
                    "2. Geometric Quality: The object appears geometrically sound and consistent across all views, with no obvious flaws like polygonal facets, broken surfaces, intersecting parts, or inconsistencies between views that indicate a poor underlying 3D model.\n"
                    "Based on your overall judgment from all views, please respond only with 'Yes' or 'No'."
                )
            elif self.cfg.prompt_type == 2:
                self.prompt_text = (
                    "Carefully evaluate the provided images, which show multiple views of a single 3D object. Does the underlying 3D object, considering all views together, meet all of the following criteria simultaneously? \n"
                    f"1. Content Match: The object corresponds to the description: {contexts}. \n"
                    "2. Geometric Quality: Based on all views combined, the object appears geometrically sound and consistent. There are no visible signs of major flaws such as multiple faces on one part (Janus-faced issue), broken surfaces, intersecting geometry, or highly unrealistic polygonal facets when considering the object from these different perspectives. \n"
                    "Strictly respond with only 'Yes' or 'No'."
                )
            elif self.cfg.prompt_type == 3:
                self.prompt_text = (
                    "Analyze the input images, which show multiple renderings/views of a single 3D object. Does the underlying 3D object, assessed from all these views, meet both of the following conditions? \n"
                    f"1. Content Match: It aligns with the description: {contexts}. \n"
                    "2. Geometric Quality: It demonstrates good geometric integrity across all views, appearing as a coherent, well-formed object without obvious visual flaws (like broken surfaces, duplicate faces, Janus issues, or severe polygonal faceting) when considered as a whole 3D shape. \n"
                    "Answer only with 'Yes' or 'No'."
                )
            elif self.cfg.prompt_type == 4:
                self.prompt_text = (
                    "Evaluate the single 3D object shown in the multiple input image views based on the following. \n"
                    "Criteria (ALL must be met based on all views collectively): \n"
                    f"1. Content Match: The object matches the text description: {contexts}. \n"
                    "2. Geometric Quality: The object's geometry appears consistent and free of major visible flaws (e.g., no Janus faces, broken surfaces, unrealistic polygons) when synthesized from all provided views. \n"
                    "Required Output: 'Yes' or 'No' ONLY."
                )
            elif self.cfg.prompt_type == 5:  # New simplified prompt
                self.prompt_text = (
                    "Carefully evaluate the 3D object shown in the multiple input image views based on the following criteria: \n"
                    f"1. Content Match: Does the object strongly match the text description: {contexts}? \n"
                    "2. Visual Plausibility: Does the object appear visually coherent and plausible across all views, without major jarring artifacts or inconsistencies? \n"
                    "Considering both criteria, answer 'Yes' only if BOTH are reasonably met. Otherwise, answer 'No'. Respond strictly with only 'Yes' or 'No'."
                )
            elif self.cfg.prompt_type == 6:  # Only text alignment
                self.prompt_text = (
                    f"Carefully evaluate the provided images, which show multiple views of a single 3D object. Does the underlying 3D object, considering all views together, correspond to the description: '{contexts}'? \n"
                    "Strictly respond with only 'Yes' or 'No'."
                )

            # messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt_text}]}]

            image_placeholders = [{"type": "image"}] * img.shape[0]
            text_element = {"type": "text", "text": self.prompt_text}
            content_list = image_placeholders + [text_element]
            messages = [{"role": "user", "content": content_list}]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            current_image_inputs = [img]
            inputs = self.processor(text=[text], images=current_image_inputs, padding=True, return_tensors="pt", do_rescale=False)
            model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

            img = self.prepare_initial_image_tensors(img, self.processor)
            processed_image = self.processor.image_processor.process_precomputed_tensors(img)

            final_inputs = {
                "input_ids": model_inputs["input_ids"], "attention_mask": model_inputs['attention_mask'],
                "pixel_values": processed_image['pixel_values'], "image_grid_thw": processed_image['image_grid_thw'],
            }

            outputs = self.vlm_model(**final_inputs)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :]
            yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
            if not self.cfg.qwen_prob_loss:
                logit_yes = last_token_logits[yes_token_id]
                logit_no = last_token_logits[no_token_id]
                reward_loss = logit_no - logit_yes
            else:
                probs = torch.softmax(last_token_logits[[yes_token_id, no_token_id]], dim=-1)
                prob_yes = probs[0]
                prob_no = probs[1]
                reward_loss = prob_no - prob_yes
            print('---------------------------------------------------------')

        return {
                "loss_sds": loss,
                'loss_qwen': reward_loss,
                "grad_norm": grad.norm(),
            }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
