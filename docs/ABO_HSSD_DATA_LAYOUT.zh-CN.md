# ABO/HSSD 数据集目录与文件格式

[English](ABO_HSSD_DATA_LAYOUT.md)

仓库只跟踪体积较小的数据清单和来源信息。图片、表面点数组、模型权重、checkpoint 和运行时回执均不入 Git。当前清单包含 8,626 条训练配对、479 条验证配对和 479 条测试配对。9 条缺少自身 UID 图片的记录被排除，没有重新分配其余数据的原始划分。

## 仓库中的文件

```text
assets/
└── native21_abo_hssd_single_image/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── rejected.jsonl
    ├── summary.json
    └── ready.json
```

每行 JSONL 描述一组图片/几何配对，二者 UID 必须一致。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `uid` | 稳定数据标识，以 `abo_` 或 `hssd_` 开头，后接源资产 SHA256。 |
| `split` | `train`、`val` 或 `test`，必须与 JSONL 文件名一致。 |
| `source` | `ABO` 或 `HSSD`。 |
| `image` | 参考 H100 工作区中选定图片的绝对路径。 |
| `image_sha256` | 选定原始 PNG 的 SHA256。 |
| `surface` | 官方 `<uid>_surface.npz` 文件的绝对路径。 |
| `sha256` | 表面数据压缩文件的 SHA256。 |
| `geometry_receipt`、`image_receipt` | 处理回执，将对应文件绑定到 UID 和源身份。 |
| `coordinate_rule` | `official_surface_npz_unit_box_no_extra_transform`。 |

仓库中的清单固定了已接受的配对关系，并保留指向参考服务器的绝对路径；它们不包含实际数据文件。迁移数据时，需保持每个 UID 的配对关系，更新路径，并使用索引构建工具重新生成 ready 校验文件。不得使用去重别名图片替代缺失的 UID 图片。

## 参考数据文件树

清单指向 Linux 工作区中的以下准备后数据：

```text
assets/abo_hssd_v1/
├── processed/<uid>/upstream_geometry_v1/
│   ├── <uid>_surface.npz
│   └── complete.json
└── renders_original/<uid>/
    ├── <selected-view>.png
    └── image_complete.json
```

部分几何回执可能位于 `processed_pilot/<uid>/upstream_geometry_v1/`；配对清单会记录实际路径。训练适配器读取官方 `<uid>_surface.npz`，其中包含 `random_surface` 和 `sharp_surface` 两个数组，每个数组形状为 `[124928, 6]`、类型为 `float16`，内容是 XYZ 坐标和法线。适配器在线无放回抽取 81,920 个点、单位化法线，并拼接官方数据接口需要的零标签，生成 `[81920, 7]`。

选中的原图是 `1024x1024` 的 RGB 或 RGBA PNG。适配器将 RGBA 合成到白色背景，将 alpha 保留为 mask，按已接受的 1.15 比例 padding，并缩放到 518 像素。单个训练样本输出 image `[3, 518, 518]` 和 mask `[1, 518, 518]`。

## 配对与就绪校验

`ready.json` 记录 split 数量、源清单哈希、仓库跟踪划分文件哈希、上游提交和坐标规则。索引校验器检查清单哈希、UID 与 split 的对应关系、数据文件大小和修改时间、回执哈希；也可以选择完整校验数据文件 SHA。训练时，数据模块会校验冻结索引，并且只读取 train split。

清单和回执只能证明文件配对有效并符合数值输入约定，不能证明相机与几何对齐，也不能证明生成形状质量。后两项需要单独验证。
