# fpga-accel

多功能图像处理模型（图像增强 / 恢复 / 超分）FPGA 加速器：一个统一可重构计算引擎，跑完整模型。

- **目标板**：iW-RainboW-G35D（XCZU19EG-2，Zynq UltraScale+）
- **实现语言**：SystemVerilog（全部新模块一律 SV；原版 Verilog 流式卷积已冻结，只作参考基线）
- **流程**：Xilinx 全命令行（vivado batch / Tcl 脚本，不用 GUI）
- **接口**：AXI4-Lite（控制/配置）、AXI4-Stream（像素流）、AXI4-MM（DDR 搬运）

## 计划文档

完整实施计划见 [docs/plan.md](docs/plan.md)（计划 v3，12 页 A4 PDF 的 Markdown 版）：

- 系统框图（加速器居中）→ §1 调研结果（去噪 / 超分 / 低光照 / 通用恢复 / YOLO / 通道数 / 卷积外算子，表 R1–R8）→ §2 设计需求（架构 + 算子集，每条可追溯到 §1）→ §3 带宽分析与模型切分 → §4 实现阶段 → §5 资源预估与风险 → §6 投入与产出 → §7 方法论与工具链 → 附：决策依据。

关键设计口径：dense 最大 11×11、DW 17×17 方核 / 31×1·1×31 条带、dilated 存储核 3–9、d≤4、等效 ≤35；通道 tiling（C_in 分块累加、C_out 分块复制）；层融合（不融合 DnCNN 1080p60 要 271 GB/s，融合后 0.81 GB/s）；前端预处理模块（降采样、1/3 通道、核 ≤5×5）。

## 目录结构

```
fpga-accel/
├── README.md
├── LICENSE
├── Makefile            # 一键复现：make sim / synth / impl / bitstream / program
├── docs/
│   ├── plan.md         # 完整实施计划（v3）
│   └── system_diagram.png
├── rtl/                # SystemVerilog 源码（参数化）
├── tb/                 # testbench + golden（Python）
├── scripts/            # Tcl 脚本（synth / impl / bitstream / program / debug / common）
├── constraints/        # XDC（时钟、引脚、时序约束）
├── sw/                 # PS 驱动 + 层调度库（C）
└── logs/               # 每次运行的日志与报告
```

## 当前状态

- **计划 v3 完成**（2026-11-19）：系统框图 + 调研结果引出设计需求。
- **阶段 0 基线完成**：SV 参数化 PX=1/2/4，dense 9×9，golden + STA + 资源已验证。
- **阶段 1–7（VM 上）**：进行中——模块设计 + 单独验证 + 模型数值验证。
- **阶段 8–11（匹兹堡，11/26 后）**：板级综合 / 集成 / 上板测试。

## 如何使用 / 复现

> 流程脚本正在补充中，以下为目标用法。

```sh
git clone <remote-url>
cd fpga-accel
make sim        # 仿真 + golden 比对（本 VM：Icarus / Yosys / OpenSTA）
make synth      # 综合（Vivado 机器）
make impl       # 实现（布局布线、时序报告）
make bitstream  # 生成 bitstream
make program    # 烧录 + ILA 抓取
make all        # 综合 → 实现 → bitstream
```

任意一台装有同版本 Vivado 的机器，`git clone` 后 `make all` 即可复现综合→实现→bitstream；Tcl 脚本开头校验 Vivado 版本号并记录到日志。

## 许可证

MIT License，见 [LICENSE](LICENSE)。
