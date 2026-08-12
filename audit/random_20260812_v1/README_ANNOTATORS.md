# 给标注专家的操作说明

## 打开页面

在项目目录 `D:\\Filez\\DownLoad\\论文3.0` 打开 PowerShell，分别启动一个专家页面：

```powershell
python scripts\\serve_manual_assignment_audit.py --audit-root experiments\\manual_worker_ppe_association_random_audit_20260812_v1 --annotator A --port 8770 --page annotation_app_zh.html
python scripts\\serve_manual_assignment_audit.py --audit-root experiments\\manual_worker_ppe_association_random_audit_20260812_v1 --annotator B --port 8771 --page annotation_app_zh.html
```

专家 A 打开 `http://127.0.0.1:8770/`，专家 B 打开 `http://127.0.0.1:8771/`。两人必须独立标注，不能互相查看答案。

## 每个框怎么标

黄色框是工人，编号是 `P1、P2...`；彩色框是 PPE，编号是 `E1、E2...`。右侧的 `E1` 卡片对应图片里的 `E1` 框。

1. 看 `E1` 实际属于哪个工人，点击对应的 `P1/P2...`。
2. 确认没有对应工人，选“无归属”。
3. 可能属于某个候选人，但遮挡、交叉或太模糊无法可靠判断，选“无法判断”。
4. 置信度：高表示清楚，中表示大致确定但有轻微遮挡，低表示勉强判断。
5. 有明显遮挡、截断、交叉或歧义时，“遮挡或歧义”选“是”，否则选“否”。
6. 不确定时不要猜，选择“无法判断”，并在备注写简短原因。

完成全部框后，先点击“导出 CSV”备份，再点击“冻结标注”。冻结后不能修改。两位专家都冻结后，才能做第三位专家仲裁。

## 严格禁止

不要打开 `sealed_proposed_assignment_reference.csv`、预测缓存、阈值结果或另一位专家的 CSV。这个审计评价 PPE 属于谁，不评价检测器有没有检测到它，也不评价工人是否安全。
