# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input

class InterleaveInferencer:
    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        device="cuda",
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = device

        model_dtype = next(self.model.parameters()).dtype

        if not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)

        self.vae_model = self.vae_model.to(device=self.device, dtype=model_dtype)
        self.vae_dtype = next(self.vae_model.parameters()).dtype

    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, self.device)
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(
        self,
        image,
        gen_context,
        vae=True,
        vit=True,
        vae_transform=None,
        vit_transform=None,
    ):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        vae_transform = vae_transform or self.vae_transform
        vit_transform = vit_transform or self.vit_transform

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = move_generation_input_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = move_generation_input_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        num_output_vae=1,
        output_raw_tensor=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens + [0] * (num_output_vae - 1),
            curr_rope=list(ropes[0] + x for x in range(num_output_vae)),
            image_sizes=[image_shape] * num_output_vae,
            new_token_ids=self.new_token_ids,
        ) 
        generation_input = move_generation_input_to_device(generation_input, self.device)
        packed_seqlens = generation_input["packed_seqlens"]
        generation_input["packed_seqlens"] = torch.sum(generation_input["packed_seqlens"], dim=0, keepdim=True, dtype=generation_input["packed_seqlens"].dtype)
        generation_input["key_values_lens"] = torch.sum(generation_input["key_values_lens"], dim=0, keepdim=True, dtype=generation_input["key_values_lens"].dtype)

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg + [0] * (num_output_vae - 1),
            curr_rope=list(ropes_cfg[0] + x for x in range(num_output_vae)),
            image_sizes=[image_shape] * num_output_vae,
        )
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, self.device)
        generation_input_cfg_text["cfg_key_values_lens"] = torch.sum(
            generation_input_cfg_text["cfg_key_values_lens"],
            dim=0,
            keepdim=True,
            dtype=generation_input_cfg_text["cfg_key_values_lens"].dtype,
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg + [0] * (num_output_vae - 1),
            curr_rope=list(ropes_cfg[0] + x for x in range(num_output_vae)),
            image_sizes=[image_shape] * num_output_vae,
        )
        generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, self.device)
        generation_input_cfg_img["cfg_key_values_lens"] = torch.sum(
            generation_input_cfg_img["cfg_key_values_lens"],
            dim=0,
            keepdim=True,
            dtype=generation_input_cfg_img["cfg_key_values_lens"].dtype,
        )


        x_0 = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            return_raw_latent=True,
        )
        unpacked_latent = x_0.split((packed_seqlens - 2).tolist())

        images = [
            self.decode_image(latent, image_shape, output_raw_tensor)
            for latent in unpacked_latent
        ]
        return images


    def decode_image(self, latent, image_shape, output_raw_tensor=False):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        latent = latent.to(dtype=self.vae_dtype)
        image = self.vae_model.decode(latent)

        if output_raw_tensor:
            image = image[0].permute(1, 2, 0).float().cpu().numpy()
        else:
            image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
            image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = move_generation_input_to_device(generation_input, self.device)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        caption=False,
        output_multiple_vae=False,
        output_raw_tensor=False,
        return_preprocessed_input=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        vae_transform=None,
        vit_transform=None,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        preprocessed_inputs = [] if return_preprocessed_input else None
        vae_transform = vae_transform or self.vae_transform
        vit_transform = vit_transform or self.vit_transform

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                    if return_preprocessed_input:
                        preprocessed_inputs.append(input_term)

                elif isinstance(input_term, Image.Image):
                    input_term = vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(
                        input_term,
                        gen_context,
                        vae=not understanding_output,
                        vae_transform=vae_transform,
                        vit_transform=vit_transform,
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    if return_preprocessed_input:
                        preprocessed_inputs.append(input_term)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think or caption:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                input_image_count = sum(
                    isinstance(item, Image.Image)
                    for item in input_lists
                )
                num_output_vae = (
                    max(input_image_count, 1)
                    if output_multiple_vae
                    else 1
                )

                imgs = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    num_output_vae=num_output_vae,
                    output_raw_tensor=output_raw_tensor,
                )

                output_list.extend(imgs)

        if return_preprocessed_input:
            return output_list, preprocessed_inputs

        return output_list
    
    def __call__(
        self, 
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            if isinstance(image, Image.Image):
                input_list.append(image)
            else:
                input_list.extend(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
