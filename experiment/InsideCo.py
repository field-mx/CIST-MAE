# -*- coding: utf-8 -*-
"""
内部聚类 (InsideCo) 实验脚本
基于高 Loss 灾难区的 Apriori 共现挖掘，寻找高度冗余/替补的内部聚类传感器。
每个聚类之内的传感器相关性最高，互相推断正确率高；
不同聚类之间相关性差，相互推断正确率低。

Usage:
    python experiment/InsideCo.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# 将项目根目录加入路径，以复用 mask_analysis.py 中的函数
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mask_analysis import loss_binning_kmeans, apriori_mining


def build_clusters_by_lift(rules, N, max_cluster_size=6, lift_threshold=1.0):
    """
    基于 Lift 排序的贪心聚类算法。
    
    逻辑：
    1. 将所有一对一关联规则去重后，按 Lift 从高到低排序。
    2. 从最高 Lift 的传感器对开始，尝试将它们放入同一个簇。
    3. 如果两个传感器都还没有分配簇 → 创建新簇。
    4. 如果其中一个已有簇且该簇未满 → 将另一个吸纳进来。
    5. 如果两个都已有簇 → 跳过（不做合并，避免雪球效应导致超大簇）。
    6. 超过 max_cluster_size 的簇不再接纳新成员。
    7. 最终没有被分配到任何簇的传感器归入"孤立组"。
    
    Args:
        rules: Apriori 关联规则 DataFrame
        N: 传感器总数
        max_cluster_size: 每个簇最多包含的传感器数
        lift_threshold: 只考虑 Lift >= 此阈值的规则
        
    Returns:
        clusters: list of dict, 每个元素 = {"sensors": [...], "avg_lift": float}
                  按 avg_lift 降序排列
    """
    # 1. 提取一对一规则，去重，按 Lift 降序
    pair_lifts = []
    seen_pairs = set()
    for _, row in rules.iterrows():
        ant = list(row['antecedents'])
        con = list(row['consequents'])
        if len(ant) == 1 and len(con) == 1:
            u = int(ant[0][1:])
            v = int(con[0][1:])
            pair = (min(u, v), max(u, v))
            if pair not in seen_pairs and row['lift'] >= lift_threshold:
                pair_lifts.append({
                    'u': pair[0], 'v': pair[1],
                    'lift': row['lift'], 'support': row['support']
                })
                seen_pairs.add(pair)
    
    # 按 Lift 降序排列
    pair_lifts.sort(key=lambda x: x['lift'], reverse=True)
    
    # 2. 贪心聚类
    sensor_to_cluster = {}  # sensor_id -> cluster_id
    clusters = []           # list of {"sensors": set(), "lift_sum": float, "lift_count": int}
    
    for pair in pair_lifts:
        u, v, lift = pair['u'], pair['v'], pair['lift']
        u_cid = sensor_to_cluster.get(u)
        v_cid = sensor_to_cluster.get(v)
        
        if u_cid is None and v_cid is None:
            # 都没有簇 → 创建新簇
            new_cid = len(clusters)
            clusters.append({"sensors": {u, v}, "lift_sum": lift, "lift_count": 1})
            sensor_to_cluster[u] = new_cid
            sensor_to_cluster[v] = new_cid
            
        elif u_cid is not None and v_cid is None:
            # u 已有簇，v 还没有 → 尝试加入 u 的簇
            if len(clusters[u_cid]["sensors"]) < max_cluster_size:
                clusters[u_cid]["sensors"].add(v)
                clusters[u_cid]["lift_sum"] += lift
                clusters[u_cid]["lift_count"] += 1
                sensor_to_cluster[v] = u_cid
                
        elif u_cid is None and v_cid is not None:
            # v 已有簇，u 还没有 → 尝试加入 v 的簇
            if len(clusters[v_cid]["sensors"]) < max_cluster_size:
                clusters[v_cid]["sensors"].add(u)
                clusters[v_cid]["lift_sum"] += lift
                clusters[v_cid]["lift_count"] += 1
                sensor_to_cluster[u] = v_cid
                
        # else: 两个都已有簇 → 跳过，不合并
    
    # 3. 收集孤立传感器（没有被分配到任何簇的）
    orphans = [i for i in range(N) if i not in sensor_to_cluster]
    
    # 4. 计算每个簇的平均 Lift 并排序
    result = []
    for c in clusters:
        avg_lift = c["lift_sum"] / c["lift_count"] if c["lift_count"] > 0 else 0
        result.append({
            "sensors": sorted(list(c["sensors"])),
            "avg_lift": avg_lift,
            "num_pairs": c["lift_count"]
        })
    
    # 按平均 Lift 降序排列
    result.sort(key=lambda x: x["avg_lift"], reverse=True)
    
    return result, orphans, pair_lifts


def main():
    save_dir = os.path.join(parent_dir, "results")
    probe_path = os.path.join(save_dir, "0.4mask_probe_results.csv")
    
    # ============ 配置参数 ============
    max_cluster_size = 6    # 每个内部簇最多包含的传感器数
    min_support = 0.3       # Apriori 最小支持度
    max_len = 2             # 频繁项集最大长度
    lift_threshold = 1.0    # 只保留 Lift >= 此值的规则
    
    print(f"\n[INFO] 读取掩码探测结果: {probe_path}")
    if not os.path.exists(probe_path):
        print(f"[ERROR] 文件不存在: {probe_path}")
        return
        
    df = pd.read_csv(probe_path)
    losses = df['avg_loss'].values
    mask_vectors = np.array([json.loads(v) for v in df['mask_vector'].values])
    N = mask_vectors.shape[1]
    
    # ============ 步骤 1: 提取灾难区 (S_high) ============
    print("\n[INFO] 使用 K-Means 分箱寻找高 Loss 灾难区 (S_high)...")
    labels, breaks = loss_binning_kmeans(losses, n_clusters=3)
    
    s_high_idx = np.where(labels == 2)[0]
    S_high_vectors = mask_vectors[s_high_idx]
    
    n_low = (labels == 0).sum()
    n_mid = (labels == 1).sum()
    n_high = (labels == 2).sum()
    print(f"  S_low:  {n_low} 组 (优质低损耗)")
    print(f"  S_mid:  {n_mid} 组")
    print(f"  S_high: {n_high} 组 (崩溃高损耗)")
    print(f"  分箱边界: {[f'{b:.6f}' for b in breaks]}")
    
    # ============ 步骤 2: Apriori 挖掘 ============
    print(f"\n[INFO] 对 S_high 进行 Apriori 关联规则挖掘 (S_high: {n_high} 组)...")
    freq_items, rules = apriori_mining(S_high_vectors, min_support=min_support, max_len=max_len)
    
    if rules.empty:
        print("[WARNING] 在支持度 0.3 下未找到强关联规则。自动降低支持度到 0.1...")
        freq_items, rules = apriori_mining(S_high_vectors, min_support=0.1, max_len=max_len)
        
    print(f"  挖掘出 {len(rules)} 条提升度 (Lift) >= 1.0 的内部关联规则。")

    # 保存频繁项集
    if len(freq_items) > 0:
        freq_save = freq_items.copy()
        freq_save['itemsets'] = freq_save['itemsets'].apply(lambda x: str(list(x)))
        freq_save.to_csv(os.path.join(save_dir, "inside_frequent_items.csv"), index=False)
        print(f"  频繁项集数量: {len(freq_items)}")
    
    # 保存关联规则
    if len(rules) > 0:
        rules_save = rules.copy()
        for col in rules_save.columns:
            if rules_save[col].dtype == object:
                rules_save[col] = rules_save[col].apply(str)
        rules_save.to_csv(os.path.join(save_dir, "inside_rules.csv"), index=False)
        print(f"  关联规则数量: {len(rules)}")
        
        # 打印 Top-30 高 Lift 传感器组合（去重）
        if 'lift' in rules.columns:
            rules_copy = rules.copy()
            rules_copy['combo'] = rules_copy.apply(
                lambda r: frozenset(r['antecedents'] | r['consequents']), axis=1)
            unique_rules = rules_copy.sort_values('lift', ascending=False).drop_duplicates(subset='combo', keep='first')
            top_rules = unique_rules.nlargest(30, 'lift')
            print(f"\n  Top-30 高 Lift 传感器组合 (去重后):")
            for _, r in top_rules.iterrows():
                ant = list(r['antecedents'])
                con = list(r['consequents'])
                ant_str = str(ant[0]) if len(ant) == 1 else str(set(ant))
                con_str = str(con[0]) if len(con) == 1 else str(set(con))
                print(f"    {ant_str} <---> {con_str}  Support={r['support']:.3f}  Lift={r['lift']:.3f}")
    
    # ============ 步骤 3: 基于 Lift 排序的贪心聚类 ============
    print(f"\n[INFO] 执行基于 Lift 排序的贪心内部聚类 (max_cluster_size={max_cluster_size})...")
    cluster_results, orphans, pair_lifts = build_clusters_by_lift(
        rules, N, max_cluster_size=max_cluster_size, lift_threshold=lift_threshold
    )
    
    # ============ 步骤 4: 输出与保存结果 ============
    print("\n" + "=" * 60)
    print("  内部聚类 (InsideCo) 划分结果")
    print("  同一簇内的传感器互为替补，相关度极高")
    print("  聚类按 avg_lift 降序排列（最强冗余组在前）")
    print("=" * 60)
    
    clusters_dict = {}
    for i, c in enumerate(cluster_results):
        sensor_names = [f"S{s}" for s in c["sensors"]]
        clusters_dict[f"Cluster_{i}"] = {
            "sensors": c["sensors"],
            "sensor_names": sensor_names,
            "avg_lift": round(c["avg_lift"], 4),
            "num_pairs": c["num_pairs"]
        }
        print(f"  内部簇 {i}: {sensor_names}  (avg_lift={c['avg_lift']:.3f}, 关联对数={c['num_pairs']})")
    
    if orphans:
        orphan_names = [f"S{s}" for s in orphans]
        clusters_dict["Orphans"] = {
            "sensors": orphans,
            "sensor_names": orphan_names,
            "avg_lift": 0,
            "num_pairs": 0
        }
        print(f"\n  孤立传感器 ({len(orphans)} 个): {orphan_names}")
    
    out_path = os.path.join(save_dir, "inside_clusters.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clusters_dict, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"[INFO] 分析完成! 结果保存至 {save_dir}/")
    print(f"  - inside_frequent_items.csv  (频繁项集)")
    print(f"  - inside_rules.csv           (关联规则)")
    print(f"  - inside_clusters.json       (内部聚类结果)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
