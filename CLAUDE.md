# rag-search 项目协作规则

## 项目信息
- 项目名：video-knowledge-rag
- GitHub：https://github.com/Nana7mi77/rag-search.git
- 主分支：main（受保护，不可直接push）

## 基本原则
- 需求先入需求表，再做研发
- 先读项目成员表，再做调度
- 派发前先判断任务类型和风险等级
- 高风险结构变化必须升级审核
- 不要新起平行实现
- 不要把前端写成半个后端
- 证据不足时明确说不确定

## 成员调度
- 调度以 `governance/tables/成员表.csv` 为主入口
- 每个成员有独立工作区（`workspaces/<member>/`）
- 成员通过分支和PR交接，不共用脏环境

## 分支策略
- `main`：稳定基线，只通过PR合并
- `feat/<member>`：各成员研发分支
- `review/<role>`：审核分支
- `fix/<req-id>`：修复型需求分支

## 默认流程
1. 需求结构化 → 需求表
2. 派发前治理 → 治理合同
3. 研发执行 → 成员工作区
4. 黑盒验证/链路采集 → verify工作区
5. 分诊 → triage
6. 合并前审核守门 → ai-review/arch-review
7. PR合并到main → release-manager
8. 回写需求表和项目日志

## 审核触发器
以下改动默认升级到ARCH REVIEW：
- 新增页面/路由
- 新增公共组件
- 新增公共 service/store/controller/agent
- 新增 API 或接口契约变化
- 删除公共能力
- 新增依赖
- 多层同时改动
- 疑似重复造轮子
- 疑似前端实现后端职责

## 工具
- codegraph：`scripts/codegraph.py` 用于代码关系分析
- push脚本：`scripts/push.sh` 用于成员提交
- pull脚本：`scripts/pull.sh` 用于拉取最新代码

## 项目表
- `governance/tables/需求表.csv`
- `governance/tables/项目日志表.csv`
- `governance/tables/成员表.csv`
