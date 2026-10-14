# 评估模式说明

所有模式使用固定的40个未见物体；每个物体三个seed。编辑模式使用相同50步Euler采样与源坐标系。源和目标没有独立归一化或ICP。

|模式|实际执行|用途|
|---|---|---|
|source|直接输出原始源网格，不通过模型|不编辑基线；判断模型是否优于复制输入|
|vae|源网格编码为Shape-VAE latent，再直接解码|检查几何压缩/重建误差；不运行DiT，不做指令编辑|
|native|官方Hunyuan3D2.1 image-to-shape，源物体第一个视角，官方图像条件，CFG5|原始生成基线；不接Qwen或指令，不是同等输入的编辑模型|
|random|预训练冻结DiT＋未训练随机MLP连接器＋源VAE latent；输入正确Qwen特征|编辑接口在没有适配训练时的基线；随机的是连接器，不是整个模型|
|connector|训练500步MLP连接器，DiT保持冻结；正确指令＋源latent|检查只训练语义连接器的贡献|
|lora|以connector_best初始化，再训练连接器和DiT注意力LoRA最多2000步；评估lora_best(step1920)|完整近似编辑方法；VLM/VAE/DiT原始主干权重冻结|
|no_source|同lora_best，移除直接拼入DiT的源VAE latent tokens|检查直接几何条件的作用；Qwen仍看源四视角，因此并非完全没有源物体信息|
|noop|同lora_best，源物体不变，将指令替换为Keep the geometry unchanged.|检查保持不变能力；summary主要对源网格评分，其余模式主要对编辑目标评分|
|shuffled|同lora_best，保留源四视角和源latent，将指令替换为manifest下一物体的指令（循环映射）|检验是否利用指令；不是打乱图片、几何或VLM token，也不是新的训练模型|

Qwen2.5-VL-7B-Instruct的条件由源四视角与指令共同前向得到，取最后一层所有有效输入token隐藏状态。它不生成文本回答。本实验没有使用闭源Buffalo VLM或官方编辑权重。

正确指令优于shuffled/noop并且对应添加/删除实际可见，才能支持指令有效性。与native相比更好、velocity loss下降、网格能导出，都不足以单独证明编辑成功。no_source为推理时去条件消融，存在分布变化；不能解释为另一个单独训练模型。
