# 大规模随机 worker-state 人工标注

这是路线 A 的独立人工状态审计包。共 220 张图像，来自 11 个 filename group，每组 20 张；共 686 个可标注人员行。抽样在查看模型结果和几何 reference 之前完成，并排除了旧的困难审计和旧的随机 PPE 归属审计。

## 启动专家 A

在仓库根目录运行：

```powershell
python scripts/serve_worker_state_audit.py `
  --audit-root audit/independent_worker_state_random_20260827_v1 `
  --annotator A --port 8795
```

然后打开 <http://127.0.0.1:8795/>。

## 启动专家 B

在另一台电脑上使用同一份审计包的独立副本，运行：

```powershell
python scripts/serve_worker_state_audit.py `
  --audit-root audit/independent_worker_state_random_20260827_v1 `
  --annotator B --port 8796
```

## B 重新独立标注

由于原 `annotator_B` 与 A 的文件完全相同，原 B 文件只保留作审计记录，不能作为独立一致性证据。请使用下面的全新空白副本重新完成：

```powershell
python scripts/serve_worker_state_audit.py `
  --audit-root audit/independent_worker_state_random_20260827_v1 `
  --annotator B_retry --port 8797
```

打开 <http://127.0.0.1:8797/>。B_retry 必须重新独立判断，不查看 `annotator_A`、原 `annotator_B` 或任何分析结果。

专家 A 和 B 必须使用不同的 `--annotator` 参数，并且不能查看对方的 CSV、模型输出、几何 reference 或阈值结果。服务器只监听本机地址，不会上传图像或答案。

## 标注规则

页面中黄色框和 `P1/P2/...` 只是帮助定位工人。对每个 P 行分别填写安全帽、反光背心和总体状态：

- `SAFE`：该项清楚可见且符合要求；
- `UNSAFE`：明确未佩戴或不符合要求；
- `REVIEW`：遮挡、太小、截断、光照不足或无法可靠判断，不要猜；
- 总体状态：任一组件明确不合格则 `UNSAFE`；两项都明确合格才 `SAFE`；否则 `REVIEW`。

置信度是专家对自己判断的把握程度，不是模型分数。若图中有明显可见但没有黄色 P 框的人员，选择“有未框出人员”并在备注说明。不要把该人员的状态套到其他 P 上。

完成全部行后先点击“导出 CSV”备份，再点击“冻结本轮”。冻结后不能修改。两位专家都冻结后，才能合并和分析；不要提前打开分析脚本生成的 reference 文件。

本审计不是 PPE 框归属审计，也不是部署安全验证。它的用途是提供不受几何候选集限制的 worker-state 人工参考。
