# 服务器逐阶段运行说明

本次仅授权编码/合成测试，以下真实命令是供你服务器逐阶段评审使用的说明，不是本地执行记录。不要一键串联所有阶段。上传完整项目（含vendor/许可证），不上传tests/_tmp。

1. 在项目内准备已有assets：官方checkpoint、既有HGS可执行。修改configs/experiment.json中的服务器source_repo、development_source、confirmation_source、historical_catalogs、checkpoint和solver_executable绝对路径。填solver_sha256与版本。服务器旧目录必须只读。配置和代码修改应在import-development之前完成。禁止自动安装/编译。
2. 环境已有Python/NumPy1.26.4/PyTorch/CUDA，目标GPU架构支持；不自动CPU回退。设置shell环境 `PYTHONDONTWRITEBYTECODE=1`、`CUDA_CACHE_DISABLE=1`；Linux可把TMPDIR设置为项目内经授权创建的临时目录。只读命令均使用-B。生产CLI没有synthetic切换。
3. 单独执行只读检查：

```bash
python -B teacher_cli.py check
python -B teacher_cli.py audit-source
python -B teacher_cli.py budget
python -B -m unittest discover -s tests -v
```

单元测试会在tests/_tmp写入，需相应授权。检查环境可用性不等于CUDA真实运行验证。audit-source必须记录UPSTREAM_BASELINE的已知差异和校准未核实，其他漂移阻断。

开发阶段（每次单独授权，默认run=runs/reference-v1，第一次必须全新目录；不能重跑已开始阶段）：

```bash
python -B teacher_cli.py import-development --start
python -B teacher_cli.py generate-dev-references --start --solver-budget 960 --solver-seconds 5 --solver-threads 1 --solver-seeds 271828 314159 161803
python -B teacher_cli.py build-dev-features --start
python -B teacher_cli.py fit-oof --start --max-fits 120
python -B teacher_cli.py assess-development --start --max-fits 5
python -B teacher_cli.py audit
```

读取assess-development/data.json：exploratory、may_continue、T*/B*、power及fallback。若development_status=DEVELOPMENT_NOT_SUPPORTED则停止，不启动新测试，不宣称普遍否定参考教师。功效不足不自动扩N；可继续预设N但报告精度限制，改变N需另行协议/预算评审。

配对校准另行授权，两侧收费1024，必须在测试前：

```bash
python -B teacher_cli.py calibrate --start --cap 1024
python -B teacher_cli.py freeze-plan --start
```

新测试各阶段另行授权；所有分数先冻结，最后B64：

```bash
python -B teacher_cli.py generate-test-states --start --cap 25600
python -B teacher_cli.py generate-test-references --start --solver-budget 1920 --solver-seconds 5 --solver-threads 1 --solver-seeds 271828 314159 161803
python -B teacher_cli.py build-test-candidates --start
python -B teacher_cli.py build-test-features --start
python -B teacher_cli.py freeze-test-predictions --start
python -B teacher_cli.py collect-test-b --start --cap 262144
python -B teacher_cli.py audit
python -B teacher_cli.py analyze
```

最后导出需写入授权及新目录（不得覆盖）：

```bash
python -B teacher_cli.py analyze --start --output runs/reference-v1/final-report
```

产生report.json/report.md，首要结论由冻结规则计算；缺失/身份错误报告INVALID，不手填成功。analyze无output只打印，不落盘。保留所有failed/started/complete与calls账本，硬中断不删锁自动重跑。传回报告、manifest、receipts和导出进行科学审查。
