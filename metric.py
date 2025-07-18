import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set

def calculate_map_at_k(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    k: int = 3,
    id_col: str = 'fname',
    pred_col: str = 'predicted_labels', # 你的预测文件中包含预测标签的列名
    gt_col: str = 'label' # 你的真实标签文件中包含真实标签的列名
) -> float:
    """
    计算 Mean Average Precision at k (MAP@k)。

    Args:
        predictions_df (pd.DataFrame): 包含模型预测结果的DataFrame。
                                       需要包含文件名列和预测标签列。
                                       预测标签列应包含字符串，多个标签用空格或逗号分隔。
                                       例如：'label1 label2 label3' 或 'label1,label2,label3'
        ground_truth_df (pd.DataFrame): 包含真实标签的DataFrame。
                                        需要包含文件名列和真实标签列。
                                        真实标签列应包含字符串，多个标签用空格或逗号分隔（如果存在多标签）。
                                        比赛说明中，train.csv的label是单标签，但实际测试集可能一个音频有多个真实标签。
                                        这里假设真实标签也可能包含多个，用逗号分隔（更通用）。
        k (int): 计算 AP@k 的 k 值。比赛中是 3。
        id_col (str): 文件名的列名。
        pred_col (str): 预测标签的列名。
        gt_col (str): 真实标签的列名。

    Returns:
        float: 计算得到的 MAP@k 值。
    """

    all_ap_scores = []
    
    # 将真实标签和预测标签统一处理成集合（Set），方便查找和比较
    # 注意：如果你的真实标签文件 (test_labels.csv) 的 'label' 列确实是单个标签字符串
    # 那么这里需要调整，例如：gt_labels_map[file_id] = {gt_row[gt_col]}
    # 但根据比赛描述 "多源声音叠加现象"，更可能一个音频有多个真实标签，
    # 所以这里假设真实标签也可以是多标签字符串（如 "label1,label2"）。
    
    # 建立真实标签的映射：文件名 -> 真实标签集合
    gt_labels_map: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        file_id = row[id_col]
        # 如果真实标签列是字符串且可能包含多个标签（如 "label1,label2"），则进行分割
        # 如果确定是单个标签，可以直接 {row[gt_col]}
        if isinstance(row[gt_col], str):
            gt_labels_map[file_id] = set(row[gt_col].replace(" ", "").split(','))
        else: # 如果是单个标签，或者其他非字符串类型
            gt_labels_map[file_id] = {str(row[gt_col])} # 确保是字符串并放入集合


    # 遍历预测结果
    num_tested_audios = 0
    for _, pred_row in predictions_df.iterrows():
        file_id = pred_row[id_col]
        
        # 确保该文件在真实标签中存在
        if file_id not in gt_labels_map:
            print(f"警告: 文件 '{file_id}' 在预测结果中，但未在真实标签中找到，跳过。")
            continue
        
        num_tested_audios += 1
        true_labels = gt_labels_map[file_id]

        # 获取预测标签，并处理成列表
        # 假设预测标签也是字符串，多个标签用空格或逗号分隔
        predicted_labels_str = str(pred_row[pred_col])
        if predicted_labels_str == "" or predicted_labels_str == "nan": # 处理空预测或NaN
            predicted_labels = []
        else:
            # 统一分隔符，以空格或逗号分隔
            predicted_labels = [
                tag.strip() for tag in predicted_labels_str.replace(',', ' ').split() if tag.strip()
            ]
        
        # 确保只取前 k 个预测
        predicted_labels = predicted_labels[:k]

        # 计算 P(k)
        precision_at_k = []
        correct_count = 0
        for i, pred_label in enumerate(predicted_labels):
            if pred_label in true_labels:
                correct_count += 1
            precision_at_k.append(correct_count / (i + 1))
        
        # 计算 AP@k
        if not precision_at_k: # 如果没有预测标签，AP为0
            ap_score = 0.0
        else:
            # 这里的 min(n, k) 中的 n 是真实标签的数量
            # 比赛公式中的 min(n, 3) 对应这里 min(len(true_labels), k)
            denominator = min(len(true_labels), k) if len(true_labels) > 0 else 1 # 避免除以0
            
            # 计算 AP@k 的求和部分
            # 只有当预测标签在真实标签中时，才累加 P(k)
            sum_pk = 0.0
            cumulative_correct = 0
            for i, pred_label in enumerate(predicted_labels):
                if pred_label in true_labels:
                    cumulative_correct += 1
                    sum_pk += (cumulative_correct / (i + 1)) # P(k)只在预测正确时计算和累加

            ap_score = sum_pk / denominator

        all_ap_scores.append(ap_score)

    if num_tested_audios == 0:
        print("没有可用于评估的音频文件。")
        return 0.0

    # 计算 MAP@k
    map_at_k = np.mean(all_ap_scores)
    return map_at_k

# --- 使用示例 ---

if __name__ == "__main__":
    # --- 1. 准备示例数据 ---
    # 模拟真实标签文件 (通常来自比赛提供的测试集标签或你的验证集标签)
    # 假设 'data_val.csv' 包含文件名和真实标签 (可能是单标签或多标签，这里用单标签演示)
    # 如果你的真实标签文件是多标签，例如 "label1,label2"，那么需要相应处理
    example_ground_truth_data = {
        'fname': ['audio_001.wav', 'audio_002.wav', 'audio_003.wav', 'audio_004.wav', 'audio_005.wav'],
        'label': ['Hi-hat', 'Saxophone', 'Trumpet', 'Cello,Violin_or_fiddle', 'Laughter'] # 示例多标签
    }
    ground_truth_df = pd.DataFrame(example_ground_truth_data)

    # 模拟预测结果文件 (你的模型输出后提交的文件)
    # 预测结果必须包含文件名和预测的标签列表（最多3个，按置信度排序）
    # 多个标签可以由空格或逗号分隔，这里使用空格分隔
    example_predictions_data = {
        'fname': ['audio_001.wav', 'audio_002.wav', 'audio_003.wav', 'audio_004.wav', 'audio_005.wav'],
        'predicted_labels': [
            'Hi-hat Snare_drum Bass_drum',     # P(1)=1/1=1, P(2)=1/2, P(3)=1/3 -> AP = (1+1/2+1/3)/min(1,3) = (1+0+0)/1 = 1  (如果只看正确项贡献P(k) = (1+0+0)/1 )
            'Cello Saxophone Trumpet',         # P(1)=0, P(2)=1/2, P(3)=1/3 -> AP = (0+1/2+0)/1 = 0.5 (如果只看正确项贡献P(k) = (0+0.5+0)/1 )
            'Trumpet Flute Oboe',              # P(1)=1, P(2)=1/2, P(3)=1/3 -> AP = (1+1/2+1/3)/1 = 1
            'Violin_or_fiddle Cello Guitar',   # P(1)=1/1, P(2)=2/2, P(3)=2/3 -> AP = (1+1+2/3)/2 = (2+2/3)/2 = 1.33/2 = 0.66 (这里n=2，分母是2)
            'Applause Squeak Laughter'         # P(1)=0, P(2)=0, P(3)=1/3 -> AP = (0+0+1/3)/1 = 0.33
            # 这里AP计算遵循实际竞赛，只累加正确预测的P(k)。
            # 例如对于 'audio_001.wav': true=['Hi-hat'], pred=['Hi-hat', 'Snare_drum', 'Bass_drum']
            # k=1: 'Hi-hat' is correct. P(1)=1/1=1. sum_pk = 1
            # k=2: 'Snare_drum' is wrong.
            # k=3: 'Bass_drum' is wrong.
            # AP = sum_pk / min(n, k) = 1 / min(1,3) = 1/1 = 1
            #
            # 对于 'audio_004.wav': true=['Cello', 'Violin_or_fiddle'], pred=['Violin_or_fiddle', 'Cello', 'Guitar']
            # n=2, k=3, min(n,k)=2
            # k=1: 'Violin_or_fiddle' is correct. P(1)=1/1=1. sum_pk = 1
            # k=2: 'Cello' is correct. P(2)=2/2=1. sum_pk = 1 + 1 = 2
            # k=3: 'Guitar' is wrong.
            # AP = sum_pk / min(n, k) = 2 / 2 = 1.0 (This implies perfect ordering for the two true labels)
            # My current AP calculation is consistent with standard MAP, but the exact formula interpretation might slightly differ depending on contest's implicit definition of P(k)
            # The contest formula for P(k) is "前k个预测中正确的数量 / k", which is standard precision.
            # The contest formula for AP@3 is "1 / min(n,3) * SUM P(k)". This sum should be over all k, but only correct predictions contribute to P(k) for that sum to be meaningful.
            # Let's adjust my AP_score calculation in the function to match standard Average Precision definition more closely,
            # which is the sum of (Precision at k) * (relevance at k) for relevant items.
            # The contest's formula is "1 / min(n,3) * SUM P(k)", which is a bit ambiguous for SUM P(k) if P(k) is always defined as correct_count/k.
            # Standard AP is sum (P_k * rel_k) / num_relevant.
            # However, for simplicity and adherence to the stated formula "SUM P(k)" over all k (1 to 3),
            # I will assume it means sum of precision@k when the k-th item is correct, divided by min(n,3).
            # No, the formula is clear: sum P(k) where P(k) is the precision at that specific k.
            # This is not standard AP. Standard AP is "sum of precision at each recall point".
            # The provided formula is: 1/min(n,3) * (P(1) + P(2) + P(3))
            # My current code uses a standard definition of AP (sum(P(k) if relevant) / num_relevant)
            # Let's re-implement AP@3 strictly based on the provided formula:
            # AP@3 = (P(1) + P(2) + P(3)) / min(n, 3)
            # where P(k) = (number of correct predictions in top k) / k
        ]
    }
    predictions_df = pd.DataFrame(example_predictions_data)

    print("\n--- 开始计算 MAP@3 ---")
    
    # 重新实现 AP@k 的计算逻辑，严格按照比赛公式 `1 / min(n,3) * Σ P(k)`
    # 这里的 Σ P(k) 是指 k=1,2,3 的 P(k) 之和
    def calculate_ap_at_3_strict(
        true_labels: Set[str],
        predicted_labels: List[str]
    ) -> float:
        pk_scores = []
        correct_count = 0
        
        # Iterate up to k=3 or the length of predicted_labels
        for i in range(min(len(predicted_labels), 3)):
            pred_label = predicted_labels[i]
            if pred_label in true_labels:
                correct_count += 1
            pk_scores.append(correct_count / (i + 1)) # P(k) = correct_at_k / k
        
        # Pad with 0s if fewer than 3 predictions
        while len(pk_scores) < 3:
            pk_scores.append(0.0) # If less than 3 predictions, P(k) for those k is 0

        sum_pk = sum(pk_scores)
        
        # Denominator: min(n, 3)
        n = len(true_labels)
        denominator = min(n, 3) if n > 0 else 1 # Handle case with no true labels to avoid division by zero

        return sum_pk / denominator


    all_ap_scores_strict = []
    num_tested_audios_strict = 0

    for _, pred_row in predictions_df.iterrows():
        file_id = pred_row[id_col]
        
        if file_id not in gt_labels_map:
            continue
        
        num_tested_audios_strict += 1
        true_labels = gt_labels_map[file_id]

        predicted_labels_str = str(pred_row[pred_col])
        if predicted_labels_str == "" or predicted_labels_str == "nan":
            predicted_labels = []
        else:
            predicted_labels = [
                tag.strip() for tag in predicted_labels_str.replace(',', ' ').split() if tag.strip()
            ]
        
        # Only consider up to top k predictions (which is 3 for AP@3)
        predicted_labels_top_k = predicted_labels[:k]

        ap_score = calculate_ap_at_3_strict(true_labels, predicted_labels_top_k)
        all_ap_scores_strict.append(ap_score)

    if num_tested_audios_strict == 0:
        final_map_score_strict = 0.0
        print("没有可用于严格评估的音频文件。")
    else:
        final_map_score_strict = np.mean(all_ap_scores_strict)

    # 最终分数
    score = final_map_score_strict * 100

    print(f"\n计算的 MAP@{k}: {final_map_score_strict:.4f}")
    print(f"最终分数 (MAP@{k} x 100): {score:.2f}")

    # 可以将上述计算函数封装为一个独立的函数
    # 示例调用
    # final_map_score = calculate_map_at_k(predictions_df, ground_truth_df, k=3)
    # print(f"使用通用函数计算的 MAP@{k}: {final_map_score:.4f}")
    # print(f"最终分数 (通用函数 MAP@{k} x 100): {final_map_score * 100:.2f}")