from dataclasses import dataclass, field
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import math
import torchvision.transforms as transforms
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        qwen_loss: float = -1.0
        qwen_prob_loss: bool = False
        prompt_type: int = 1
        sds_prompt_not_none: bool = True
        vlm_prompt: str = "An orange and a red apple are on a plate, a green apple is beside the plate, all on a table."

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")
        print('DreamFusionDiffusionGuidance Config', self.cfg)
        self.call_counter = 0

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        if self.cfg.qwen_loss > 0.0:
            ckpt_path = "/home/pub/bwm/models/Qwen2.5-VL-7B-Instruct"
            self.processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
            self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            self.vlm_model.eval().requires_grad_(False).to(self.device)
            print('Loading Qwen Model', ckpt_path, self.cfg.qwen_loss)
            # --- 添加以下代码来查看 image_processor 信息 ---
            try:
                # 检查 self.processor 是否有 image_processor 属性且不为 None
                if hasattr(self.processor, 'image_processor') and self.processor.image_processor is not None:
                    img_processor_obj = self.processor.image_processor
                    img_processor_class = img_processor_obj.__class__
                    img_processor_name = img_processor_class.__name__

                    # 尝试获取定义该类的文件路径
                    try:
                        img_processor_file_path = inspect.getfile(img_processor_class)
                    except TypeError:
                        # 对于某些内置类型或动态生成的类，可能无法获取文件路径
                        img_processor_file_path = "N/A (无法获取文件路径)"

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
            # --- 查看 image_processor 信息代码结束 ---

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        latents_denoised = (latents_noisy - sigma * noise_pred) / alpha
        image_denoised = self.decode_latents(latents_denoised)

        grad = w * (noise_pred - noise)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, grad_img, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    # smart_resize (保持不变)
    def smart_resize(self, height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056):
        if height < factor or width < factor:
            print(f"警告: Height ({height}) 或 Width ({width}) 小于 factor ({factor}). 可能需要调整 factor.")
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
        # 确保返回整数
        h_bar = int(max(factor, h_bar))
        w_bar = int(max(factor, w_bar))
        # 再次检查是否能被factor整除（ceil/floor后可能不是）
        h_bar = round(h_bar / factor) * factor
        w_bar = round(w_bar / factor) * factor
        return int(h_bar), int(w_bar)

    # prepare_initial_image_tensors (保持不变)
    def prepare_initial_image_tensors(self, image_tensor, processor):
        img_processor = processor.image_processor
        patch_size = img_processor.patch_size
        merge_size = img_processor.merge_size
        h_orig, w_orig = image_tensor.shape[-2], image_tensor.shape[-1]
        target_h, target_w = self.smart_resize(
            h_orig, w_orig, factor=patch_size * merge_size,
            min_pixels=img_processor.min_pixels, max_pixels=img_processor.max_pixels
        )
        # print(f"计算出的兼容目标尺寸 H={target_h}, W={target_w}")
        # if target_h == 0 or target_w == 0:
        #      raise ValueError(f"计算出的目标尺寸为零 H={target_h}, W={target_w}。请检查 smart_resize 或输入图像。")

        img_transforms = transforms.Compose([
            transforms.Resize([target_h, target_w], interpolation=img_processor.resample, antialias=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean=img_processor.image_mean, std=img_processor.image_std)
        ])
        image1_tensor_norm = img_transforms(image_tensor)
        # print(f"初始图片已调整大小至 ({target_h}, {target_w}), 并标准化。")
        # print(f"设备: {image1_tensor_norm.device}, Requires Grad: True")
        # print(f"Tensor 1 Shape: {image1_tensor_norm.shape}")
        return image1_tensor_norm

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        # <<< Increment call counter >>>
        self.call_counter += 1

        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils, elevation, azimuth, camera_distances
            )
            grad_img = torch.tensor([0.0], dtype=grad.dtype).to(grad.device)
        else:
            grad, grad_img, guidance_eval_utils = self.compute_grad_sds(
                latents,
                rgb_BCHW_512,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
            )

        grad = torch.nan_to_num(grad)

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_sds_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
            )
            guidance_out["loss_sds_img"] = loss_sds_img

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        reward_loss = torch.tensor(0.0, device=latents.device)  # Initialize reward loss
        if self.cfg.qwen_loss > 0.0 and self.call_counter <= 10000:
            # img = pred_rgb[0].unsqueeze(0).clamp(0, 1)
            print('rgb_BCHW_512', rgb_BCHW_512.shape) # torch.Size([1, 3, 512, 512])
            img = rgb_BCHW_512.clamp(0, 1)
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
                print(f"计算得到的 Logit('Yes'): {logit_yes.item():.4f}")
                print(f"计算得到的 Logit('No'): {logit_no.item():.4f}")
                print(f"原始损失 (Logit('No') - Logit('Yes')): {reward_loss.item():.4f}")
            else:
                probs = torch.softmax(last_token_logits[[yes_token_id, no_token_id]], dim=-1)
                prob_yes = probs[0]
                prob_no = probs[1]
                reward_loss = prob_no - prob_yes
                print(f"计算得到的 Prob('Yes'): {prob_yes.item():.4f}")
                print(f"计算得到的 Prob('No'): {prob_no.item():.4f}")
                print(f"概率损失 (Prob('No') - Prob('Yes')): {reward_loss.item():.4f}")
            print('---------------------------------------------------------')
            guidance_out["loss_qwen"] = reward_loss

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
