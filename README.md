# 固定 Neural Repair 的参考解信息价值诊断

本包交付代码与实验协议；**没有真实研究结果**。参考解用于特权教师信息价值诊断，不是可部署学生，更不证明完整 NLNS 搜索收益。

入口：`python -B teacher_cli.py check`。只读命令不创建目录、缓存或 SQLite；所有写入阶段显式 `--start`，真实 Repair 另需 `--cap`，拟合需 `--max-fits`，求解需显式三种子/线程/时间/总预算。不得自动安装、下载、编译或补采。

阅读顺序：PROTOCOL.md → FEATURE_SPEC.md → SOURCE_AUDIT.md → BUDGET.md → SERVER_RUNBOOK.md。故障排查见 TROUBLESHOOTING.md，验证证据见 VALIDATION.md。

依赖：合成性质验证需 NumPy；服务器真实后端严格要求 NumPy 1.26.4、支持目标 CUDA 的 PyTorch 和已安装 HGS。无 sklearn/pandas 依赖。项目不安装这些依赖。Python 3.10+；服务器建议复用经过确认的原环境。

旧核心原样存放 vendor/，来源与原始字节 SHA-256 在 vendor/PROVENANCE.json，GPLv3 许可证在 vendor/LICENSE。新工程复用其特征、模型预测、CART、VRP 和 actor/Repair；旧路径检查不修改。新工程路径与身份控制在 teacher_select/core.py，生产数据不能走 synthetic 入口。

合成测试：`python -B -m unittest discover -s tests -v`。仅工程测试允许临时拟合与合成数据，证据不能进入生产报告。临时目录固定在项目 tests/_tmp 内。不要将其与 125 次真实研究拟合混为一谈。

本地只完成合成验证和静态/只读来源审查。运行资产不随代码复制：checkpoint、旧开发导出、历史 catalog、HGS 可执行文件由服务器已有材料配置。完整协议冻结前编辑配置；真实阶段开始后代码与配置身份漂移会阻断。
