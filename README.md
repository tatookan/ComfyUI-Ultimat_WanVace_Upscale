# ComfyUI-SuperUltimateVaceTools
超究VACE工具，一些基于万象2.1 VACE模型的Comfyui插件工具，尝试以更好的方式实现VACE的视频生成/编辑功能。  
SuperUltimateVaceTools, some Comfyui custom nodes for wan2.1 VACE, attempt to implement VACE video generation/editing in a better way.  
包含以下插件：  
Including following nodes:  

- 超究视频放大 | Super Ultimate Vace Upscale
- 超究长视频 | Super Ultimate VACE Long Video
- 更多功能待添加 | pending to add more...

## SuperUltimateVaceUpscale
对视频进行分割放大，支持空间分割以及时间分割。  
Upscale video by splitting it into tiled areas, supports spatial tiling and temporal tiling.  

https://github.com/user-attachments/assets/ba9f91a1-b097-4ff2-9780-db85d014c978

### 空间分割 | Spatial tiling
默认情况下视频将由左向右、由上向下平均分割成矩形，分割块的数目由放大后尺寸与生成尺寸共同决定。块之间叠加区域同样由这些数值决定，为了避免出现明显分割痕迹，放大尺寸对生成尺寸取余的数值不能过小。  
如果不满足于默认的分割方案，可以尝试把`Custom Crop Area`节点连接到`croparea_list`，它有更多特殊的分割方案。  
By default, the video will be evenly divided into rectangles from left to right and top to bottom, with the number of segmentation blocks determined by both the upscale dimensions and the generate dimensions. The overlapping area between blocks is also determined by these values. In order to avoid obvious segmentation marks, the remainder value of upscaled to generated should not be too little.  
If you are not satisfied with the default segmentation plan, you can try connecting the `Custom Crop Area` node to the `croarea_ist`, which has more special segmentation plans.

### 时间分割 | Temporal tiling
当输入视频帧数比`length`设置值更大时，将启用时间分割，将视频分割为多个部分分别进行放大，最后拼接在一起。为了避免分割时间点出现跳变，需要设置适当的`crossfade_frame`作为过渡区域。  
如果输入视频是循环视频，希望放大后仍然保持首尾间无缝循环，应当设置适当的`loopback_crossfade`作为循环过渡区域。  
When the input video frame count is more than the `length` value, temporal tiling will be enabled, dividing the video into multiple parts for upscale, and finally splicing them together. To avoid sudden change in the segmentation time points, it is necessary to set an appropriate `crossfade-frame` as the transition part.   
If the input video is a looped video and you want to maintain seamless looping between the beginning and end after upscale, you should set an appropriate `loopback_crossfade` as the transition part for the loop.

### 参考和控制 | Reference and control
为了获得更加稳定的结果，建议使用参考图片和控制视频。参考图片最好是输入视频的第一帧的高清放大版本（因此本节点适合用于你用一张高清图片进行i2v结果的放大）  
当使用参考图片后，启用`crop_ref`将按照分割视频的方式分割参考图片，并在每个分割部分重绘时使用分割后的参考图进行该区域参考。启用`ref_as_init_frame`后会直接用参考图替换为视频的第一帧，并以此为参考指导各区域重绘生成。  
对于控制视频，VACE支持许多控制方法，你可以尽情尝试不同的控制方式带来的区别。
To get more controlable result, it is recommended to use reference image and control video. It is recommended that the reference image is a high-definition and detailed version of the first frame of the input video (therefore, this node is suitable for the situation when you are using a high-resolution image to do i2v and then upscale the i2v result)  
After using the reference image, enabling 'crop-ref' will divide the reference image according to the video segmentation plan, and use the segmented reference image for the region reference when denoising each segmented part. If enable 'ref_as_init_frame', the first frame of the video will be directly replaced with the reference image, and this will be used as a reference to guide the following frames denoising of each region.   
For control video, VACE supports many control methods, and you can freely try the differences brought by different control methods.

## 安装 | Install
方法1：在`ComfyUI\custom_nodes`路径下执行命令  
Way 1: run following cmd command at the path `ComfyUI\custom_nodes`

    git clone https://github.com/bbaudio-2025/ComfyUI-SuperUltimateVaceTools

方法2：下载本项目并解压缩到`ComfyUI\custom_nodes`  
Way 2: Clone this repo into `custom_nodes` folder.  
