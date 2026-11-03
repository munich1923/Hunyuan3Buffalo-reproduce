# ABO/HSSD 原始图片迁移

[English](ABO_HSSD_ORIGINAL_IMAGES.md)

此 CPU 数据准备流程为每个几何合格的 UID 选择并复制一张 TRELLIS 原始条件 PNG。几何仍使用官方处理结果；此流程没有证明图像相机与 VAE 坐标对齐。

## 身份与视角约定

请求会固定候选清单 SHA、迁移清单 SHA 和既有 trellis_piflow 图片索引 SHA。每条源记录必须与数据集、原始 SHA256 和 `file_identifier` 相符。原始 GLB 不会再次完整哈希；流程检查已完成迁移回执和当前文件大小/修改时间。

TRELLIS 相机到世界矩阵使用以原点为中心的 Z-up 场景。仰角为 `atan2(z, hypot(x,y))`，方位角为 `atan2(y,x)`。仰角范围含端点 `[-60,60]` 度，方位角不限。`camera_angle_x` 是水平视场角，不是仰角。相机矩阵必须是有限、仿射、正确朝向的刚体变换，视场角必须在 `(0, pi)` 内。

候选 PNG 必须能完整解码为原始 `1024x1024 RGBA`。文件按原始文件名排序，再使用由 `SHA256([20261027, "original-render-view-v1", UID])` 派生的本地随机种子，在有效帧中均匀选择一张。不会改变全局、caption、噪声、posterior 或时间随机数生成器。该策略只在有限可用视图中均匀抽样，不代表按仰角均匀抽样。图片保持原始字节，不转换 RGB，也不缩放。

历史索引缺图、源身份无效、变换矩阵异常、PNG 缺失或没有合格视图都会被明确记录；不会换对象、放宽角度或重新渲染。即便缺图，对象仍留在原始数据划分中。

## 产物与阶段

脚本位于 `scripts/native_21dit/Abo_Hssd_datapre/`，包括 `render_images.py`、`render_images_relay.py` 和 `test_render_images.py`。独立证据目录是 `results/abo_hssd_v1/render_images_v1`。图片和 `paired_index.jsonl` 位于 `assets/abo_hssd_v1/renders_original`。

Windows relay 直接从 H100 加载代码，并在现有 SSH 连接之间以内存传输二进制数据；不会在本地写图片或凭证。源扫描和复制期间，本机和网络必须保持可用。

阶段依次为：准备来源请求；最多两个线程扫描源目录；等待所有几何任务完成；必要时用 CPU manifest builder 固定去重和划分；冻结合格图片选择；传输 ABO 与 HSSD 各四个试验对象；解码、SHA 和重启复用检查；传输剩余图片；完整核对图片、角度、SHA、划分和几何回执。

流程不会修改或启动已有 GPU 几何控制器。图片步骤使用 CPU，manifest 子进程设置 `CUDA_VISIBLE_DEVICES` 为空。已有几何进程只观察，不会被自动重启；若它在产生最终回执前退出，图片流程会停止以便诊断。

## 续传与安全约定

不要启动额外的 relay 或 receiver。先检查 `relay_status.json`、`status.json`、进程命令行和失败日志。receiver 持有 `transfer.lock`。目标资产先写入每 UID 的临时目录，流式校验 PNG 与完整 `transforms.json` 的 SHA；图片解码和回执提交完成后才重命名。缺少回执的最终目录或发生变化的冻结合同会导致失败关闭。中断的临时文件可重试，已完成且匹配的文件可复用。源扫描记录可在源元数据和帧文件状态指纹匹配时复用；不同观察会保留在 `source_scan_history`。

`source_selection_inventory.json` 是候选预检查，不是最终合格图片集。只有几何和不可变划分完成后生成的 `frozen_selection.json` 决定传输对象。`paired_index.jsonl` 包含每个可用合格 UID、其划分及去重别名代表 UID；它不会改动 `train.jsonl`、`val.jsonl`、`test.jsonl` 或几何回执。

## 验收

CPU 测试覆盖正负角度与边界、视场角区别、刚体矩阵校验、确定性选择和全局随机状态隔离、身份不匹配、损坏 PNG、无合格帧、过早冻结拒绝、中断二进制传输、相同回执复用和目标变化拒绝。完整复制前，8 个真实试验资产必须通过解码、原图 SHA、相机和重启复用检查。

`final_verification.json` 分开报告已复制文件、几何覆盖率、缺失源图、按来源/划分计数和负载大小。`migration_complete.json` 只代表图片迁移完成，不代表数据整体就绪。原始相机与归一化记录会保留，`camera_alignment_to_vae` 明确标为 `not established`。

实现验证记录为 11 项 CPU 测试通过，包括别名继承原划分和缺图保持划分。源读取预检查使用两个线程；曾观测到的源进程 VmHWM 为 115508 kB，本地 relay 工作集为 34930688 字节。这些是当时观测值，不是峰值保证。
