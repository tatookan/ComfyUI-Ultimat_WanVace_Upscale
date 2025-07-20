import torch
# from PIL import Image, ImageOps, ImageSequence
import comfy.samplers
import comfy.sample
import nodes
import node_helpers
import latent_preview
from comfy.comfy_types import IO
from comfy.utils import common_upscale

def emptyimage(width, height, batch_size=1, color=(0,0,0)):
    r = torch.full([batch_size, height, width, 1], color[0] / 255, dtype=torch.float32, device="cpu")
    g = torch.full([batch_size, height, width, 1], color[1] / 255, dtype=torch.float32, device="cpu")
    b = torch.full([batch_size, height, width, 1], color[2] / 255, dtype=torch.float32, device="cpu")
    return torch.cat((r, g, b), dim=-1)

def imagecrop(image, width, height, x, y):
    x = min(x, image.shape[2] - 1)
    y = min(y, image.shape[1] - 1)
    to_x = width + x
    to_y = height + y
    img = image[:,y:to_y, x:to_x, :]
    return img

def feather(mask, left=0, top=0, right=0, bottom=0):
    # from comfyui
    output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()

    left = min(left, output.shape[-1])
    right = min(right, output.shape[-1])
    top = min(top, output.shape[-2])
    bottom = min(bottom, output.shape[-2])

    for x in range(left):
        feather_rate = (x + 1.0) / left
        output[:, :, x] *= feather_rate

    for x in range(right):
        feather_rate = (x + 1) / right
        output[:, :, -x] *= feather_rate

    for y in range(top):
        feather_rate = (y + 1) / top
        output[:, y, :] *= feather_rate

    for y in range(bottom):
        feather_rate = (y + 1) / bottom
        output[:, -y, :] *= feather_rate

    return output

def repeat_tensor(tensor, batch, dim=0):
    repeat_list = []
    for n in range(batch):
        repeat_list.append(tensor)
    result = torch.cat(repeat_list, dim=dim)
    return result

def imgcomposite(destination, source, x, y, mask):
    des_copy = destination.clone()
    des_crop = des_copy[:, y:(source.shape[1] + y), x:(source.shape[2] + x), :]
    composed_area = des_crop * (1 - mask.unsqueeze(-1)) + source * mask.unsqueeze(-1)
    des_copy[:, y:(source.shape[1]+y), x:(source.shape[2]+x), :] = composed_area
    return des_copy

def maskasemble(batchsize, width, height, value_bg, value_fg, left, top, right, bottom):
    output = torch.full((batchsize, height, width), value_bg, dtype=torch.float32, device="cpu")
    output[:, 0:top, :] = value_fg
    output[:, (height - bottom):height, :] = value_fg
    output[:, :, 0:left] = value_fg
    output[:, :, (width - right):width] = value_fg
    return output

def add_noise(image, noise_aug_strength, seed):
    # from KJNODES
    torch.manual_seed(seed)
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * noise_aug_strength
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image_out = image + image_noise
    return image_out

def spatialistgen(width_upscale, height_upscale, width, height, spatial_multiplier=16):
    if width >= width_upscale or height >= height_upscale:
        raise ValueError("spatialistgen: 放大尺寸应该大于生成尺寸\ndimension_upscale should be large than dimension")
    width = width // spatial_multiplier * spatial_multiplier
    height = height // spatial_multiplier * spatial_multiplier
    num_tile_x =  width_upscale // width + 1
    num_tile_y = height_upscale // height + 1
    pad_x = (num_tile_x * width - width_upscale) // (num_tile_x - 1)
    pad_x_res = (num_tile_x * width - width_upscale) % (num_tile_x - 1)
    pad_y = (num_tile_y * height - height_upscale) // (num_tile_y - 1)
    pad_y_res = (num_tile_y * height - height_upscale) % (num_tile_y - 1)

    croparea_list = []
    for i in range(num_tile_y):
        for j in range(num_tile_x):
            croparea_list.append({
                'width_crop': width,
                'height_crop': height,
                'offset_x': j * width - (pad_x * j + pad_x_res if j == num_tile_x - 1 else pad_x * j),
                'offset_y': i * height - (pad_y * i + pad_y_res if i == num_tile_y - 1 else pad_y * i),
                'mask_left': 0 if j == 0 else pad_x + pad_x_res if j == num_tile_x - 1 else pad_x,
                'mask_right': 0,
                'mask_top': 0 if i == 0 else pad_y + pad_y_res if j == num_tile_y - 1 else pad_y,
                'mask_bottom': 0,
                'feather_left': 0 if j == 0 else pad_x + pad_x_res if j == num_tile_x - 1 else pad_x,
                'feather_right': 0,
                'feather_top': 0 if i == 0 else pad_y + pad_y_res if i == num_tile_y - 1 else pad_y,
                'feather_bottom': 0,
            })
    return croparea_list

def temporalistgen(num_total_frame, length, num_crossfade, num_loopback_crossfade, temporal_multiplier=4):
    if num_total_frame < length:
        raise ValueError("temporalistgen: 视频帧数应该大于或等于length\nframe count of input video should be larger than or equal to length")
    res_frame = num_total_frame
    slice_list = []
    while res_frame + num_crossfade > length:
        start_index = num_total_frame - res_frame - num_crossfade if res_frame != num_total_frame else 0
        res_frame = res_frame - length
        slice_list.append({
            'start_index': start_index,
            'length': length,
            'num_crossfade': num_crossfade,
            'flag_final_slice': False,
        })
    res_frame += num_crossfade
    if (res_frame - 1) % temporal_multiplier != 0:
        res_frame_padded = res_frame + temporal_multiplier - (res_frame - 1) % temporal_multiplier
    else:
        res_frame_padded = res_frame
    num_crossfade_end = res_frame_padded - res_frame + num_crossfade
    if num_loopback_crossfade > num_crossfade_end:
        raise ValueError("temporalistgen: num_loopback_crossfade过大，尝试减小\nnum_loopback_crossfade is too large, try to decrease it")
    slice_list.append({
        'start_index': num_total_frame - res_frame_padded,
        'length': res_frame_padded,
        'num_crossfade': num_crossfade_end,
        'flag_final_slice': True,
    })
    return slice_list

def crossfadevideos(video1, video2, num_corssfade_frame):
    if video1.ndim != video2.ndim:
        raise ValueError("crossfadevideos: 拼接图片类型不一致\nImageType Mismatch")
    if video1[[0],].shape != video2[[0],].shape:
        raise ValueError("crossfadevideos: 拼接图片尺寸不一致\nImageSize Mismatch")
    if num_corssfade_frame > video1.shape[0] or num_corssfade_frame > video2.shape[0]:
        raise ValueError("crossfadevideos: 拼接图片数目应大于过渡数目\nVideoLength should be longer than CrossLength")
    video_slice1 = video1[:-num_corssfade_frame]
    video_slice2 = video1[-num_corssfade_frame:]
    video_slice3 = video2[:num_corssfade_frame]
    video_slice4 = video2[num_corssfade_frame:]
    alpha_list = []
    count = num_corssfade_frame + 1
    while count > 1:
        alpha_list.append((count - 1) / (num_corssfade_frame + 1))
        count -= 1
    alpha_list.reverse()
    blend_list = []
    index = 0
    for alpha in alpha_list:
        mixed = video_slice2[[index],] * (1 - alpha) + video_slice3[[index],] * alpha
        blend_list.append(mixed)
        index += 1
    blended_slice = torch.cat(blend_list, dim=0)
    result = torch.cat((video_slice1, blended_slice, video_slice4), dim=0)
    return result

def vace_sample(model, positive, negative, vae, width, height, length, strength, seed, cfg, sampler_name, scheduler, steps, denoise, video,
                control_video=None, control_masks=None, reference_image=None, tile_control_video=None):
    # from comfyui
    latent_length = ((length - 1) // 4) + 1
    if control_video is not None:
        control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if control_video.shape[0] < length:
            control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
    else:
        control_video = torch.ones((length, height, width, 3)) * 0.5

    if reference_image is not None:
        reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        reference_image = vae.encode(reference_image[:, :, :, :3])
        reference_image_vaed = reference_image.clone()
        reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

    if control_masks is None:
        mask = torch.ones((length, height, width, 1))
    else:
        mask = control_masks
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
        if mask.shape[0] < length:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

    control_video = control_video - 0.5
    inactive = (control_video * (1 - mask)) + 0.5
    reactive = (control_video * mask) + 0.5

    inactive = vae.encode(inactive[:, :, :, :3])
    reactive = vae.encode(reactive[:, :, :, :3])
    control_video_latent = torch.cat((inactive, reactive), dim=1)
    if reference_image is not None:
        control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask = mask.permute(2, 4, 0, 1, 3)
    mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

    trim_latent = 0
    if reference_image is not None:
        mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
        mask = torch.cat((mask_pad, mask), dim=1)
        latent_length += reference_image.shape[2]
        trim_latent = reference_image.shape[2]

    mask = mask.unsqueeze(0)

    positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

    # sample

    latent = vae.encode(video[:,:,:,:3])
    if reference_image is not None:
        latent = torch.cat((reference_image_vaed, latent), dim=2)
    noise = comfy.sample.prepare_noise(latent, seed)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                                denoise=denoise, disable_noise=None, start_step=None, last_step=None,
                                force_full_denoise=False, noise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples[:, :, trim_latent:]
    images = vae.decode(samples)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images

class UltimateVideoUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Only VACE models are supported"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "input_video": ("IMAGE", ),
                "width_upscale": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height_upscale": ("INT", {"default": 720, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "pad_mask_limit": ("INT", {"default": 32, "min": 8, "max": 512, "step": 8}),
                "crossfade_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "loopback_crossfade": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "crop_ref": ("BOOLEAN", {"default": False}),
                "ref_as_init_frame": ("BOOLEAN", {"default": False}),
                "noise_aug": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step":0.001, "round": 0.001, }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, }),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "croparea_list": ("LIST", ),
                "reference_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("video", )
    OUTPUT_TOOLTIPS = ("Upscaled Video",)
    FUNCTION = "upscale_video"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = """
视频分块放大|Upscale video by splitting into tiled areas
by bbaudio
联系方式
QQ：1953761458
Email：1953761458@qq.com
QQ群：948626609
"""
    
    def upscale_video(self, model, width_upscale, height_upscale, width, height, length, pad_mask_limit, crossfade_frame, loopback_crossfade, 
                      crop_ref, ref_as_init_frame, noise_aug, input_video, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, vae, 
                      croparea_list=None, reference_image=None, control_video=None):
        if control_video is not None and control_video.shape[0] != input_video.shape[0]:
            raise ValueError("控制视频帧数与输入视频帧数应当一致\nFrame count of ControlVideo and InputVideo should be the same")
        if loopback_crossfade > 0:
            cross_slice = input_video[:loopback_crossfade].clone()
            input_video = torch.cat((input_video, cross_slice), dim=0)
            if control_video is not None:
                control_cross_slice = control_video[:loopback_crossfade].clone()
                control_video = torch.cat((control_video, control_cross_slice), dim=0)
        total_frame = input_video.shape[0]
        if total_frame > length and crossfade_frame == 0:
            raise ValueError("视频帧数大于length，需要设置crossfade_frame以启用时间分割\nFrame count of input video is larger than length, need set a proper value for crossfade_frame to enable temporal tiling")        
        strength = 1 # VACE Strength
        temporalist = temporalistgen(total_frame, length, crossfade_frame, loopback_crossfade)
        upscaled_videos_list = []
        turn_index = 0
        for turn in temporalist:
            start_index = turn['start_index']
            length_n = turn['length']
            cross_fade = turn['num_crossfade']
            upscaled_video = comfy.utils.common_upscale(input_video[start_index:(start_index + length_n)].movedim(-1, 1), width_upscale, height_upscale, "bilinear", "center").movedim(1, -1)
            if control_video is not None:
                up_scaled_control = comfy.utils.common_upscale(control_video[start_index:(start_index + length_n)].movedim(-1, 1), width_upscale, height_upscale, "bilinear", "center").movedim(1, -1)
            upscaled_video = add_noise(upscaled_video, noise_aug, seed)
            if turn_index > 0 and cross_fade > 0:
                upscaled_video[:cross_fade] = upscaled_videos_list[turn_index-1][-cross_fade:].clone()
            if croparea_list is None:
                croparea_list = spatialistgen(width_upscale, height_upscale, width, height)
            result_video = torch.full((length_n, height_upscale, width_upscale, 3), 0.5, device='cpu')
            index = 0
            for item in croparea_list:
                width_crop_n = item['width_crop']
                height_crop_n = item['height_crop']
                offset_x_n = item['offset_x']
                offset_y_n = item['offset_y']
                video = imagecrop(upscaled_video, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                mask_ctl = maskasemble(1, width_crop_n, height_crop_n, 1, 0, 
                                    min(item['mask_left'], pad_mask_limit), 
                                    min(item['mask_top'], pad_mask_limit), 
                                    min(item['mask_right'], pad_mask_limit), 
                                    min(item['mask_bottom'], pad_mask_limit))
                mask_ctl = repeat_tensor(mask_ctl, length_n)
                if turn_index > 0 and cross_fade > 0:
                    mask_ctl[:cross_fade] = torch.full((cross_fade, height_crop_n, width_crop_n,), 0.0, device='cpu')
                crop_gen = imagecrop(result_video, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                if control_video is not None:
                    crop_ctl = imagecrop(up_scaled_control, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    controls = imgcomposite(crop_ctl, crop_gen, 0, 0, 1-mask_ctl) if index != 0 else crop_ctl
                else:
                    controls = crop_gen[:length_n] if index != 0 else torch.full((length_n, height_crop_n, width_crop_n, 3), 0.5, device='cpu')
                if reference_image is not None:
                    reference_image = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width_upscale, height_upscale, "bilinear", "center").movedim(1, -1)
                    if crop_ref is True:
                        refimg = imagecrop(reference_image, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    else:
                        refimg = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width_crop_n, height_crop_n, "bilinear", "center").movedim(1, -1)
                    if ref_as_init_frame is True and turn_index == 0:
                        init_ref = imagecrop(reference_image, width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                        controls[:1,:,:,:] = init_ref
                        mask_ctl[:1,:,:] = torch.full((1, height_crop_n, width_crop_n), 0.0, device='cpu')
                else:
                    refimg = None
                if turn_index > 0 and cross_fade > 0:
                    init_ctl = imagecrop(upscaled_videos_list[turn_index-1][-cross_fade:].clone(), width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    controls[:cross_fade] = init_ctl
                    mask_ctl[:cross_fade] = torch.full((1, height_crop_n, width_crop_n), 0.0, device='cpu')
                if turn['flag_final_slice'] is True and loopback_crossfade > 0:
                    end_ctl = imagecrop(upscaled_videos_list[0][:loopback_crossfade].clone(), width_crop_n, height_crop_n, offset_x_n, offset_y_n)
                    controls[-loopback_crossfade:] = end_ctl
                    mask_ctl[-loopback_crossfade:] = torch.full((1, height_crop_n, width_crop_n), 0.0, device='cpu')
                sampled_video = vace_sample(model, positive, negative, vae, width_crop_n, height_crop_n, length_n, strength, seed, cfg, sampler_name, scheduler, steps, denoise, video,
                    controls, mask_ctl, refimg)
                mask_feather = feather(torch.full((1, height_crop_n, width_crop_n), 1.0, device='cpu'), item['feather_left'], item['feather_top'], item['feather_right'], item['feather_bottom'])
                mask_feather = repeat_tensor(mask_feather, length_n)
                result_video = imgcomposite(result_video, sampled_video, offset_x_n, offset_y_n, mask_feather)
                index += 1
                total_tile = len(temporalist) * len(croparea_list)
                print('第', turn_index + 1, '部分视频第', index, '块生成完成；整体完成', 100*(turn_index * len(croparea_list) + index)/total_tile, '%')
            upscaled_videos_list.append(result_video)
            turn_index += 1
        result_video = upscaled_videos_list.pop(0)
        index = 0
        while index < len(upscaled_videos_list):
            cross_fade = temporalist[index + 1]['num_crossfade']
            result_video = crossfadevideos(result_video, upscaled_videos_list[index], cross_fade)
            index += 1
        if loopback_crossfade > 0:
            crossed_start = crossfadevideos(result_video[-loopback_crossfade:], result_video[:loopback_crossfade], loopback_crossfade)
            result_video[:loopback_crossfade] = crossed_start
            result_video = result_video[:(total_frame - loopback_crossfade)]
        return (result_video, )

class CustomCropArea:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width_upscale": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height_upscale": ("INT", {"default": 720, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "presets": (["'H' for wide screen", "'三' for long narrow screen"], {
                    "default": "'H' for wide screen",
                    "tooltip": "预设分割方案\nPresets of cropping plan"
                }),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "custom_croplist_gen"
    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = "Use preset of cropping plan"
    def custom_croplist_gen(self, width_upscale, height_upscale, presets):
        if presets == "'H' for wide screen":
            if height_upscale%16 != 0:
                raise ValueError("‘H’方案下放大高度必须为16的倍数\n'H' plan requires height_upscale to be multiplier of 16")
            result = [
                {
                'width_crop': width_upscale//2 + 16 - width_upscale//2%16 + 32,
                'height_crop': height_upscale//2 + 16 - height_upscale//2%16 + 16,
                'offset_x': width_upscale//4 - 32,
                'offset_y': 0,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale//2 + 16 - width_upscale//2%16 + 32,
                'height_crop': height_upscale//2 + 16 - height_upscale//2%16 + 16,
                'offset_x': width_upscale//4 - 32,
                'offset_y': height_upscale - (height_upscale//2 - height_upscale//2%16 +32),
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 32,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 32,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale//4 + 16 - width_upscale//2%16 + 16,
                'height_crop': height_upscale,
                'offset_x': 0,
                'offset_y': 0,
                'mask_left': 0,
                'mask_right': 32,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 32,
                'feather_top': 0,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale//4 + 16 - width_upscale//2%16 + 16,
                'height_crop': height_upscale,
                'offset_x': width_upscale- (width_upscale//4 + 16 - width_upscale//2%16 + 16),
                'offset_y': 0,
                'mask_left': 32,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 32,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': 0,
                },
            ]
        elif presets == "'三' for long narrow screen":
            if width_upscale%16 != 0:
                raise ValueError("‘三’方案下放大宽度必须为16的倍数\n'三' plan requires height_upscale to be multiplier of 16")
            pad = 64
            h_i = (height_upscale + 3 * pad)//4
            h_res = h_i % 16
            h = h_i - h_res
            result = [
                {
                'width_crop': width_upscale,
                'height_crop': h,
                'offset_x': 0,
                'offset_y': h - pad,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale,
                'height_crop': h,
                'offset_x': 0,
                'offset_y': 0,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': 0,
                'mask_bottom': pad,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': 0,
                'feather_bottom': pad,
                },
                {
                'width_crop': width_upscale,
                'height_crop':h ,
                'offset_x': 0,
                'offset_y': 2 * h - 2 * pad,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': pad,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': pad,
                'feather_bottom': 0,
                },
                {
                'width_crop': width_upscale,
                'height_crop': height_upscale - (3 * h - 3 * pad) - (height_upscale - 3 * h + 3 * pad)%16 + 16,
                'offset_x': 0,
                'offset_y': 3 * h - 3 * pad + (height_upscale - 3 * h + 3 * pad)%16 - 16,
                'mask_left': 0,
                'mask_right': 0,
                'mask_top': pad,
                'mask_bottom': 0,
                'feather_left': 0,
                'feather_right': 0,
                'feather_top': pad,
                'feather_bottom': 0,
                },
            ]
        return (result,)

class RegionalBatchPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "prompt_list": ("STRING", ),
                "croparea_list": ("LIST", ),
            }
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("croparea_list", )
    FUNCTION = "func"
    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = "batch conditioning prompt list"
    def func(self, clip, prompt_list, croparea_list):
        if len(prompt_list) != len(croparea_list):
            raise ValueError("提示词队列长度与切割队列长度不一致，检查节点连接是否正确\nLength of prompt_list is not same as croparea_list, check nodes connection")
        index = 0
        for prompt in prompt_list:
            tokens = clip.tokenize(prompt)
            croparea_list[index]['cond_p'] = clip.encode_from_tokens_scheduled(tokens)
            index += 1

        return (croparea_list, )

# VaceLongVideo
def sort_list(vace_control_list):
    n = len(vace_control_list)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if vace_control_list[j]['frame_position'] > vace_control_list[j + 1]['frame_position']:
                vace_control_list[j], vace_control_list[j + 1] = vace_control_list[j + 1], vace_control_list[j]
                swapped = True
        if not swapped:
            break
    return vace_control_list

def check_overlap(vace_control_list):
    n = len(vace_control_list)
    for i in range(n):
        for j in range(i + 1, n):
            if vace_control_list[i]['frame_position'] == vace_control_list[j]['frame_position']:
                raise ValueError("控制帧位置重复，检查控制帧位置设置\nidentical frame_position detected, check the frame_position setting")
            elif vace_control_list[j]['frame_position'] <= vace_control_list[i]['control_end_index'] and vace_control_list[j]['control_end_index'] >= vace_control_list[i]['frame_position']:
                raise ValueError("控制帧区域重叠，检查控制帧位置设置\nsome control frames overlapped, check the frame_position setting")
    return None

class VaceLongVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Only VACE models are supported"}),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "loopback_crossfade": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "vace_prompt_list": ("PROMPTLIST", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, }),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "vace_control_list": ("CONTROLIMAGELIST", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    OUTPUT_TOOLTIPS = ("Generated Video",)
    FUNCTION = "long_video"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = """
VACE长视频拼接|Long video by concating multiple parts
by bbaudio
联系方式
QQ：1953761458
Email：1953761458@qq.com
QQ群：948626609
"""
    def long_video(self, model, width, height, loopback_crossfade, vace_prompt_list, seed, steps, cfg, sampler_name, scheduler, 
                   denoise, vae, vace_control_list=None):
        # check prompt list
        if len(vace_prompt_list) == 1:
            vace_prompt_list[0]['init_crossfade_frame'] = 0
        # get total frame
        total_frame = 0
        for item in vace_prompt_list:
            num_frame = item['num_frame']
            init_crossfade_frame = item['init_crossfade_frame']
            total_frame += num_frame - init_crossfade_frame
            if loopback_crossfade > num_frame:
                raise ValueError("循环过渡帧数目不能超过生成长度\nloopback_crossfade can not be larger than length of generation")
        # deal with control list
        control_video = torch.full((total_frame, height, width, 3), 0.5, device='cpu')
        control_mask = torch.full((total_frame, height, width), 1.0, device='cpu')
        if vace_control_list is not None:
            check_overlap(vace_control_list)
            vace_control_list = sort_list(vace_control_list)
            index_final = vace_control_list[-1]['control_end_index']
            if index_final >= total_frame:
                raise ValueError("控制帧长度超过生成长度\nLength of control image exceeds length of generation")
            for item in vace_control_list:
                index_start = item['frame_position']
                index_end = item['control_end_index']
                control_images = comfy.utils.common_upscale(item['control_image'].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                control_images = repeat_tensor(control_images, item['repeat'])
                custom_mask = item['custom_mask']
                if custom_mask is not None:
                    if custom_mask.shape[1] != height or custom_mask.shape[2] != width:
                        custom_mask = common_upscale(custom_mask.unsqueeze(1), width, height, "nearest-exact").squeeze(1)
                control_video[index_start:index_end + 1] = control_images[:index_end + 1 - index_start]
                if custom_mask is None and item['masked'] is True :
                    control_mask[index_start:index_end + 1] = torch.full((index_end + 1 - index_start, height, width), 0.0, device='cpu')
                elif custom_mask is not None and item['masked'] is True :
                    control_mask[index_start:index_end + 1] = custom_mask
        # deal with prompt list
        sampled = []
        vace_prompt_list[-1]['flag_end'] = True
        processed_frame_count = 0
        for item in vace_prompt_list:
            num_frame = item['num_frame']
            cond_p = item['cond_p']
            cond_n = item['cond_n']
            init_crossfade_frame = item['init_crossfade_frame']
            refine_init = item['refine_init']
            ref_image = item['ref_image']
            # control
            if processed_frame_count == 0:
                controls = control_video[:num_frame].clone()
                mask_ctl = control_mask[:num_frame].clone()
            else:
                controls = control_video[processed_frame_count - init_crossfade_frame:processed_frame_count - init_crossfade_frame + num_frame].clone()
                mask_ctl = control_mask[processed_frame_count - init_crossfade_frame:processed_frame_count - init_crossfade_frame + num_frame].clone()
                controls[:init_crossfade_frame] = sampled[-1][-init_crossfade_frame:] * (1 - refine_init) + torch.full((init_crossfade_frame, height, width, 3), refine_init, device='cpu')
                if refine_init < 0.01:
                    mask_ctl[:init_crossfade_frame] = torch.full((init_crossfade_frame, height, width), 0.0, device='cpu')
                else:
                    mask_ctl[0] = torch.full((1, height, width), 0.0, device='cpu')
                    mask_ctl[1:init_crossfade_frame] = torch.full((init_crossfade_frame - 1, height, width), 1.0, device='cpu')
            if item['flag_end'] is True and loopback_crossfade > 0:
                controls[-loopback_crossfade:] = sampled[0][:loopback_crossfade].clone()
                mask_ctl[-loopback_crossfade:] = torch.full((loopback_crossfade, height, width), 0.0, device='cpu')
            empty_video = torch.zeros([num_frame, height, width, 3])
            sample_result = vace_sample(model, cond_p, cond_n, vae, width, height, num_frame, 1, seed, cfg, sampler_name, scheduler, steps, denoise, empty_video,
                    controls, mask_ctl, ref_image)
            processed_frame_count += num_frame - init_crossfade_frame
            sampled.append(sample_result)
        result_video = sampled.pop(0)
        index = 0
        while index < len(sampled):
            cross_fade = vace_prompt_list[index + 1]['init_crossfade_frame']
            result_video = crossfadevideos(result_video, sampled[index], cross_fade)
            index += 1
        if loopback_crossfade > 0:
            crossed_start = crossfadevideos(result_video[-loopback_crossfade:], result_video[:loopback_crossfade], loopback_crossfade)
            result_video[:loopback_crossfade] = crossed_start
            result_video = result_video[:(total_frame - loopback_crossfade)]
        return (result_video, )

class VACEControlImageCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_image": ("IMAGE", ),
                "frame_position": ("INT", {"default": 0, "min": 0, "max": 65535, "step": 1}),
                "masked": ("BOOLEAN", {"default": False}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 65535, "step": 1}),
            },
            "optional": {
                "custom_mask": ("MASK", ),
                "previous_control": ("CONTROLIMAGELIST", ),
            }
        }

    RETURN_TYPES = ("CONTROLIMAGELIST", )
    RETURN_NAMES = ("vace_control_list", )
    FUNCTION = "combine_controls"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def combine_controls(self, control_image, frame_position, masked, repeat, custom_mask=None, previous_control=None):
        control_list = []
        if previous_control is not None:
            control_list.extend(previous_control)
        control_end_index = frame_position + control_image.shape[0] - 1 + (repeat - 1)
        control_list.append({
            'frame_position': frame_position,
            'control_end_index': control_end_index,
            'control_image': control_image,
            'custom_mask': custom_mask,
            'masked': masked,
            'repeat': repeat,
        })
        return (control_list, )

class VACEPromptCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_frame": ("INT", {"default": 81, "min": 5, "max": 65535, "step": 4}),
                "init_crossfade_frame": ("INT", {"default": 3, "min": 0, "max": 65535, "step": 1}),
                "refine_init": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "ref_image": ("IMAGE", ),
                "previous_prompt": ("PROMPTLIST", ),
            }
        }

    RETURN_TYPES = ("PROMPTLIST", )
    RETURN_NAMES = ("vace_prompt_list", )
    FUNCTION = "combine_prompt"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def combine_prompt(self, clip, positive_prompt, negative_prompt, num_frame, init_crossfade_frame, refine_init, ref_image=None, previous_prompt=None):
        if init_crossfade_frame > num_frame:
            raise ValueError("过渡帧数目不能大于总帧数\ninit_crossfade_frame can not be larger than num_frame")
        prompt_list = []
        if previous_prompt is not None:
            prompt_list.extend(previous_prompt)
        p_tokens = clip.tokenize(positive_prompt)
        n_tokens = clip.tokenize(negative_prompt)
        cond_p =  clip.encode_from_tokens_scheduled(p_tokens)
        cond_n = clip.encode_from_tokens_scheduled(n_tokens)
        prompt_list.append({
            'cond_p': cond_p,
            'cond_n': cond_n,
            'num_frame': num_frame,
            'init_crossfade_frame': init_crossfade_frame,
            'refine_init': refine_init,
            'ref_image': ref_image,
            "flag_end": False
        })
        return (prompt_list, )

class VACEPromptCheckTotalFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_list": ("PROMPTLIST", ),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("total_frame", )
    FUNCTION = "check_total_frame"

    CATEGORY = "SuperUltimateVaceTools"
    DESCRIPTION = ""
    def check_total_frame(self, prompt_list):
        total_frame = 0
        for item in prompt_list:
            total_frame += item['num_frame'] - item['init_crossfade_frame']
        return (total_frame, )

NODE_CLASS_MAPPINGS = {
    "SuperUltimateVACEUpscale": UltimateVideoUpscaler,
    "CustomCropArea": CustomCropArea,
    "RegionalBatchPrompt": RegionalBatchPrompt,
    "VACEControlImageCombine": VACEControlImageCombine,
    "VACEPromptCombine": VACEPromptCombine,
    "VaceLongVideo": VaceLongVideo,
    "VACEPromptCheckTotalFrame": VACEPromptCheckTotalFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperUltimateVACEUpscale": "SuperUltimate VACE Upscale",
    "CustomCropArea": "Custom Crop Area",
    "RegionalBatchPrompt": "Batch Prompt Crop Area",
    "VACEControlImageCombine": "VACE Control Image Combine",
    "VACEPromptCombine": "VACE Prompt Combine",
    "VaceLongVideo": "SuperUltimate VACE Long Video",
    "VACEPromptCheckTotalFrame": "Check Total Frame",
}
