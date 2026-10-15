# 停止和失败排查

先区分工程INVALID、探索性开发未支持、确认STOP_PRACTICAL和统计INCONCLUSIVE。没有真实数据时不得宣布教师有/无价值。

| 阻断 | 只读排查 | 处理 |
|---|---|---|
| Source/config identity drift | 对比manifest.code与源码/配置哈希 | 不改已开始协议；保留原run，另行评审 |
| UPSTREAM_BASELINE已知差异 | audit-source记录expected/actual | 仅该来源记录差异可继续，核心漂移阻断 |
| 权重/设备缺失 | check、服务器既有环境 | 不下载、安装或CPU回退 |
| Stage already started/active.lock | 最近50条关键calls、started/failed/complete | 保留失败历史；不自动删锁/续跑/补预算 |
| 校准成本不一致 | 相同语义seed的single/homogeneous成本 | 阻断真实测试，不修actor/mask、删除异常或重采 |
| HGS无可行输出 | 三个固定种子exit/status/runtime | 不追加，保留实例并统一回退 |
| HGS成本/客户/精度错误 | 输入round-trip、depot、容量、重算成本 | INVALID，不报告研究负结果 |
| 数据不完整或混批padding | 逐行JSONL与batch身份、repeat coverage | INVALID，不丢例、不补采 |
| 开发无增量 | 探索性OOF独立V选择收益 | 停止本版本新测试 |
| t/bootstrap矛盾 | 主比较区间与drop1/drop2 | INCONCLUSIVE，不选有利子组 |

日志只按关键字筛选最近50条；不要全读GPU日志/SQLite/缓存。calls.jsonl为append-only，started事件在调用前fsync；无terminal视为不确定，计入cap保守上界。不得用删除记录绕过预算。已开始阶段不可复用输出；恢复计划须单独评审及冻结剩余预算，不由本CLI自动提供。
