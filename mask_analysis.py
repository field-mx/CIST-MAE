# -*- coding: utf-8 -*-
"""
步骤 B：Loss 分箱 + Apriori 频繁项集挖掘

Usage:
    python mask_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    import jenkspy
    HAS_JENKSPY = True
except ImportError:
    HAS_JENKSPY = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

# loss分类算法*2
def loss_binning_jenks(losses, n_classes=3):
    """Jenks Natural Breaks 一维分箱"""
    breaks = jenkspy.jenks_breaks(losses.tolist(), n_classes=n_classes)
    labels = np.zeros(len(losses), dtype=int)
    for i, v in enumerate(losses):
        if v <= breaks[1]:
            labels[i] = 0  # S_low
        elif v <= breaks[2]:
            labels[i] = 1  # S_mid
        else:
            labels[i] = 2  # S_high
    return labels, breaks


def loss_binning_kmeans(losses, n_clusters=3):
    """K-Means 一维分箱"""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_raw = km.fit_predict(losses.reshape(-1, 1))
    # 按聚类中心排序：中心最小 -> S_low(0), 中间 -> S_mid(1), 最大 -> S_high(2)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    label_map = {order[i]: i for i in range(n_clusters)}
    labels = np.array([label_map[l] for l in labels_raw])
    breaks = [losses.min()] + sorted(centers.tolist()) + [losses.max()]
    return labels, breaks

# 自定义多路交互 Lift 计算函数
def compute_lift_rules(freq_items, min_threshold=1.0):
    """
    根据频繁项集计算多路交互 Lift（提升度）。
    
    算法流程：
    1. 构建支持度查找表：将所有频繁项集的 support 存入字典
    2. 遍历所有长度 >= 2 的频繁项集
    3. 计算每个频繁项集的多路 Lift = Support(A∪B∪C) / (Support(A) * Support(B) * Support(C))
    4. 只保留 Lift >= min_threshold 的频繁项集
    
    输入:
        freq_items: apriori 返回的 DataFrame，包含 'itemsets' 和 'support' 两列
        min_threshold: Lift 的最小阈值（默认 1.0，即只保留正向协同）
    
    输出:
        rules_df: DataFrame，包含 itemsets, support, lift 三列
                  每个频繁项集只产生一条记录（无前件/后件之分）
    """
    # 步骤 1：构建单传感器支持度查找表 {frozenset({Si}): support}
    single_support = {}
    for _, row in freq_items.iterrows():
        if len(row['itemsets']) == 1:
            single_support[row['itemsets']] = row['support']
    
    # 步骤 2：遍历所有长度 >= 2 的频繁项集，计算多路 Lift
    rules_list = []
    for _, row in freq_items.iterrows():
        itemset = row['itemsets']
        itemset_support = row['support']
        
        if len(itemset) < 2:
            continue  # 单传感器不需要计算协同
        
        # 步骤 3：计算分母 = 各单传感器支持度的乘积
        denominator = 1.0
        all_found = True
        for sensor in itemset:
            key = frozenset([sensor])
            if key not in single_support:
                all_found = False
                break
            denominator *= single_support[key]
        
        if not all_found or denominator == 0:
            continue
        
        # 多路 Lift = Support(全集) / ∏ Support(各单元素)
        lift = itemset_support / denominator
        
        # 步骤 4：只保留满足阈值的
        if lift >= min_threshold:
            rules_list.append({
                'itemsets': itemset,
                'support': itemset_support,
                'lift': lift,
            })
    
    rules_df = pd.DataFrame(rules_list) if rules_list else pd.DataFrame()
    return rules_df


# 寻找sensor之间的绑定关系
def apriori_mining(mask_vectors, min_support=0.1, max_len=4):
    """
    对 S_low 集合的掩码向量进行 Apriori 频繁项集挖掘。
    寻找 "哪些传感器同时为 1（可见）时，大概率会落入低 Loss 区"。
    """
    if not HAS_MLXTEND:
        print("[WARNING] mlxtend 未安装，使用简化版频率统计替代 Apriori")
        return simple_frequency_analysis(mask_vectors, min_support)

    N = mask_vectors.shape[1]
    col_names = [f"S{i}" for i in range(N)]
    df_bool = pd.DataFrame(mask_vectors.astype(bool), columns=col_names)

    freq_items = apriori(df_bool, min_support=min_support, use_colnames=True, max_len=max_len)

    if len(freq_items) == 0:
        print("[WARNING] 未找到满足最小支持度的频繁项集，尝试降低阈值...")
        # 返回1 support=组合在低loss测试中出现的百分比
        # 返回2 itemsets = 该组合的sensor序号
        freq_items = apriori(df_bool, min_support=min_support * 0.5, use_colnames=True, max_len=3)

    if len(freq_items) > 0:
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
    else:
        rules = pd.DataFrame()

    return freq_items, rules


def simple_frequency_analysis(mask_vectors, min_support=0.3):
    """简化版：计算单传感器和传感器对的频率"""
    N = mask_vectors.shape[1]
    K = mask_vectors.shape[0]
    
    # 单传感器支持度
    single_support = mask_vectors.mean(axis=0)
    
    # 传感器对支持度
    pair_results = []
    for i in range(N):
        for j in range(i + 1, N):
            joint = (mask_vectors[:, i] * mask_vectors[:, j]).mean()
            expected = single_support[i] * single_support[j]
            lift = joint / expected if expected > 0 else 0
            if joint >= min_support:
                pair_results.append({
                    'sensor_A': f'S{i}', 'sensor_B': f'S{j}',
                    'support_joint': joint,
                    'support_A': single_support[i],
                    'support_B': single_support[j],
                    'lift': lift,
                })
    
    freq_items = pd.DataFrame({
        'itemsets': [frozenset([f'S{i}']) for i in range(N)],
        'support': single_support,
    })
    rules = pd.DataFrame(pair_results) if pair_results else pd.DataFrame()
    return freq_items, rules


def compute_sensor_importance(freq_items, rules, N):
    """
    根据 Apriori 结果计算每个传感器的保留概率。
    
    策略：
    - 在高 Lift 规则中频繁出现的传感器 → 高保留概率（黄金传感器）
    - 在频繁项集中几乎不出现的传感器 → 低保留概率（冗余传感器）
    """
    # 初始化：所有传感器默认中等重要性
    importance = np.ones(N) * 0.5
    
    # 1. 基于单传感器支持度调整
    for _, row in freq_items.iterrows():
        items = row['itemsets']
        if len(items) == 1:
            sensor_name = list(items)[0]
            idx = int(sensor_name[1:])
            # 单传感器支持度高 → 在 S_low 中经常可见 → 重要
            importance[idx] = max(importance[idx], 0.3 + row['support'] * 0.6)
    
    # 2. 基于关联规则中的 Lift 进一步提升（去重：同一组合只计算一次）
    if len(rules) > 0 and 'lift' in rules.columns:
        seen_combos = set()
        for _, row in rules.iterrows():
            lift = row['lift']
            if lift > 1.2:  # 显著正相关
                # 合并前件和后件，得到完整的传感器组合
                combo = frozenset(row['antecedents'] | row['consequents'])
                if combo in seen_combos:
                    continue  # 同一组合已经加过分了，跳过
                seen_combos.add(combo)
                sensors = list(combo)
                for s in sensors:
                    if isinstance(s, str) and s.startswith('S'):
                        idx = int(s[1:])
                        boost = min(0.2, (lift - 1.0) * 0.1)
                        importance[idx] = min(0.95, importance[idx] + boost)
    
    # 3. 将重要性映射为保留概率
    # 归一化到 [0.1, 0.9] 范围
    if importance.max() > importance.min():
        retain_probs = 0.1 + 0.8 * (importance - importance.min()) / (importance.max() - importance.min())
    else:
        retain_probs = np.ones(N) * 0.5
    
    return retain_probs, importance


def run_analysis():  
    """主函数：运行分析流水线"""
    save_dir = "results"
    probe_path = os.path.join(save_dir, "0.4mask_probe_results.csv")
    # 优loss中最小出现频率
    min_support=0.5
    # 最大组合长度
    max_len = 4
    
    print(f"[INFO] 读取掩码探测结果: {probe_path}")
    df = pd.read_csv(probe_path)
    losses = df['avg_loss'].values
    mask_vectors = np.array([json.loads(v) for v in df['mask_vector'].values])
    N = mask_vectors.shape[1]
    K = len(df)
    
    # ============ 步骤 B-2: 一维分箱 ============
    print(f"\n[INFO] 一维分箱 (K={K} 个样本)...")
    if HAS_JENKSPY:
        print("  使用 Jenks Natural Breaks")
        labels, breaks = loss_binning_jenks(losses)
    else:
        print("  使用 K-Means 分箱 (jenkspy 未安装)")
        labels, breaks = loss_binning_kmeans(losses)
    
    n_low = (labels == 0).sum()
    n_mid = (labels == 1).sum()
    n_high = (labels == 2).sum()
    print(f"  S_low:  {n_low} 组 (优质低损耗)")
    print(f"  S_mid:  {n_mid} 组")
    print(f"  S_high: {n_high} 组 (崩溃高损耗)")
    print(f"  分箱边界: {[f'{b:.6f}' for b in breaks]}")
    
    # 保存分箱结果
    df['bin_label'] = labels
    df['bin_name'] = df['bin_label'].map({0: 'S_low', 1: 'S_mid', 2: 'S_high'})
    df.to_csv(os.path.join(save_dir, "mask_binning.csv"), index=False)
    
    # ============ 步骤 B-3: Apriori 挖掘 ============
    print(f"\n[INFO] Apriori 频繁项集挖掘 (S_low: {n_low} 组)...")
    low_mask_vectors = mask_vectors[labels == 0]
    
    freq_items, rules = apriori_mining(low_mask_vectors, min_support, max_len)
    
    # 保存频繁项集
    if len(freq_items) > 0:
        freq_save = freq_items.copy()
        freq_save['itemsets'] = freq_save['itemsets'].apply(lambda x: str(list(x)))
        freq_save.to_csv(os.path.join(save_dir, "apriori_frequent_items.csv"), index=False)
        print(f"  频繁项集数量: {len(freq_items)}")
    
    # 保存关联规则
    if len(rules) > 0:
        rules_save = rules.copy()
        for col in rules_save.columns:
            if rules_save[col].dtype == object:
                rules_save[col] = rules_save[col].apply(str)
        rules_save.to_csv(os.path.join(save_dir, "apriori_rules.csv"), index=False)
        print(f"  关联规则数量: {len(rules)}")
        # 打印 Top-50 高 Lift 传感器组合（去重：同一组合只显示一次）
        if 'lift' in rules.columns:
            # 为每条规则生成组合标识，取同一组合中最大的 Lift
            rules_copy = rules.copy()
            rules_copy['combo'] = rules_copy.apply(
                lambda r: frozenset(r['antecedents'] | r['consequents']), axis=1)
            # 按组合去重，保留最大 Lift
            unique_rules = rules_copy.sort_values('lift', ascending=False).drop_duplicates(subset='combo', keep='first')
            top_rules = unique_rules.nlargest(50, 'lift')
            print(f"\n  Top-50 高 Lift 传感器组合 (去重后):")
            for _, r in top_rules.iterrows():
                ant = list(r['antecedents'])
                con = list(r['consequents'])
                ant_str = str(ant[0]) if len(ant) == 1 else str(set(ant))
                con_str = str(con[0]) if len(con) == 1 else str(set(con))

                print(f"    {ant_str} -> {con_str}  Support={r['support']:.3f}  Lift={r['lift']:.3f}")
    
    # ============ 计算传感器重要性 ============
    print(f"\n[INFO] 计算传感器保留概率...")
    retain_probs, importance = compute_sensor_importance(freq_items, rules, N)
    
    sensor_df = pd.DataFrame({
        'sensor_id': [f'S{i}' for i in range(N)],
        'importance': importance,
        'retain_prob': retain_probs,
        'category': ['golden' if p >= 0.6 else ('redundant' if p <= 0.25 else 'normal') 
                      for p in retain_probs],
    })
    sensor_df = sensor_df.sort_values('retain_prob', ascending=False)
    sensor_df.to_csv(os.path.join(save_dir, "sensor_importance.csv"), index=False)
    
    n_golden = (sensor_df['category'] == 'golden').sum()
    n_redundant = (sensor_df['category'] == 'redundant').sum()
    n_normal = (sensor_df['category'] == 'normal').sum()
    
    print(f"  黄金传感器 (retain >= 0.6): {n_golden} 个")
    print(f"  普通传感器: {n_normal} 个")
    print(f"  冗余传感器 (retain <= 0.25): {n_redundant} 个")
    
    if n_golden > 0:
        golden = sensor_df[sensor_df['category'] == 'golden']
        print(f"\n  黄金传感器列表:")
        for _, row in golden.iterrows():
            print(f"    {row['sensor_id']}: retain_prob={row['retain_prob']:.3f}")
    
    print(f"\n{'='*50}")
    print(f"[INFO] 分析完成! 结果保存至 {save_dir}/")
    # loss分箱结果
    print(f"  - mask_binning.csv")
    # 出现频率分出的频繁集项结果（support）
    print(f"  - apriori_frequent_items.csv")
    # 相关性检验（lift）
    print(f"  - apriori_rules.csv")
    # 传感器重要度结果（retain_prob）
    print(f"  - sensor_importance.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_analysis()
