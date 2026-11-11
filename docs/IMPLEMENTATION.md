# 论文与代码对应关系

依据 2026-11-11 工作区当前主稿及补充材料；源码 SHA-256 记录在 `manuscript_sources.json`。历史 `main.visual.tex/main.updated.tex` 不作为主实现依据。

| 论文公式/机制 | 实现 |
| --- | --- |
| pipeline、global-state、local-size、local-state、guidance | `RestorationModel.forward`、`RGLDSE.forward` |
| residual-target、state-loss | `state_loss` |
| coarse-field、affine、correction | `DSPC.forward` |
| joint、B4/B5 coarse loss | `RestorationModel.loss` |
| output-projection、bounded-projection | `project_affine` |
| error-decomposition、correction-utility、oracle-strength | `analyze_arrays`、`oracle_strength` |
| residual shift / projected shift | `analyze_arrays`、`projected_shift` |
| bounded interpolation and strength model | `gate_training`、`restore_gate`、`evaluate_model` |

## 关键约束

- 主网络只接收退化输入；训练中的 key 是同一退化图像的另一个独立裁剪。query/GT 共享裁剪及增强。
- RGL-DSE 的两个注入标量为 tanh(alpha/beta)，初始均为零。
- 局部状态先分别预测，双线性插值后拼接；监督目标在对应尺度平均池化。
- DSPC 对 `[X,Y0]` 平均池化，先插值 log-gain 后 exp。输出层为零初始化，训练不裁剪。
- 平均字段对照平均的是 exp 后的 g。独立共享头必须重新训练，不能把同次前向的 Y0 标成独立 B0。
- B2 注册但不调用 64→16 的直接状态投影；总新增参数 181,324。只有实际调用的参数才计入计算路径；激活参数也受消融开关影响。
- 仿射 LS 的目标为 C。Box-LS 枚举内部解和四个边界最小值，并处理常量通道；无约束投影的正交恒等式不套用于 Box-LS。
- 门控训练冻结主网络，24 维特征按 RGB 成组，顺序为 Y0 mean/std、C−Y0 mean/RMS、g mean/std、b mean/std。标准差使用 population 定义。
- 门控的输出是一张图的一个标量，不是逐像素强度图；推理绝不调用 oracle。
- 理论误差恒等式在未裁剪 RGB 空间计算；PSNR/SSIM 使用独立评估路径。

## 论文未完整指定的实现选择

以下是为使代码可执行而明确补全的选择，不代表找回了原实验配置。

1. **局部分支卷积 bias**：三个 64→32 的 3×3 卷积及融合分支两个 3×3 卷积使用 `bias=False`；三个 32→1 的 1×1 头保留 bias。这样局部分支恰为 93,987 参数，符合论文参数表。若全部启用 bias，会多出 224 参数。该选择由参数表推定。
2. Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0；workers=0 便于跨平台验证。联合阶段从第一步到最后一步覆盖余弦端点。
3. 默认每 500 步验证、每任务取清单中前 128 条；清单顺序须事先固定。编码器保留末次权重；仅联合阶段按恢复分数选模。
4. 逐样本轮询 rain/haze/snow 后有放回抽图，独立的 draw ID 决定增强。两阶段重置抽样流，种子相差固定常数。每个默认完整阶段三任务总预算严格相同。
5. 评估裁剪锚定左上角；要求原输入与 GT 已对齐同尺寸。推理则保留原整图尺寸。没有 padding、TTA 或 tiling。
6. SSIM 窗口与 covariance 明确设定，但缺少历史评估程序，不能声称与旧分数逐位一致。
7. 单调曲线节点默认 `[0,1]` 等距 17 点，采用非负节点增量和 BVLS 求解。颜色矩阵为每图全局 3×3+偏移最小二乘。二者均不读 GT；默认曲线节点并非已恢复的历史节点。
8. 缓存中验证端点保留 float64；训练 K/T/Q 与神经网络以 float32 计算。门控最后一层权重/偏置为零，初始 λ=0.5。
9. 冻结 B2 的缓存校验记录所属 split；它不自动修复原场景标识，也不能通过路径检查识别内容相同但被重新命名的图片。
10. 非有限 PSNR（完全相同图像的正无穷）在 JSON 中明确写为字符串 `"inf"`，避免写出非法 JSON。用于配对统计时须先处理此类退化案例。

## 外部骨干与验证范围

`OfficialAirNet` 依据作者源码的实际接口：训练 CBDE 返回 `(z,logits,labels,F)`，推理返回 `(z,F)`，DGRN 接收 `(X,G)`。官方 DGRN 仍由第三方代码提供，不在本工程中用普通卷积伪装替代。

参考来源：[AirNet 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html)、[作者实现](https://github.com/XLearning-SCU/2022-CVPR-AirNet)、[MMCV 安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。第三方代码与权重遵循各自许可，本目录不重新声明其授权。

额外 Restormer/PromptIR 训练配方、完整外部数据集下载、受控雾数据生成、论文全部表格的一键重跑、运行时 GPU 性能测量不包含在当前实现内。DSPC 是独立模块，可通过 `DSPC()(X, other_backbone(X))` 接入其他图像恢复骨干，具体训练需使用对应真实配方。

没有原始权重、清单及运行环境时，不能由论文汇总成绩推出新代码必然得到相同结果，也不会将合成测试输出或文章中的数值填写为本次测量。
