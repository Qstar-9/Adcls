import os
import pandas as pd
from sklearn.model_selection import train_test_split # 导入 train_test_split

def create_custom_audio_list_stratified_split(
    csv_filepath: str,
    audio_base_dir: str,
    output_list_dir: str,
    test_size: float = 0.2, # 测试集比例，默认为 0.2 (20%)
    random_state: int = 10 # 随机种子，用于复现性
):
    """
    根据给定的CSV元数据和音频目录，以分层抽样的方式生成训练集、测试集和标签列表文件。

    Args:
        csv_filepath (str): 包含文件名和标签的CSV文件的路径。
        audio_base_dir (str): 音频文件所在的根目录。
        output_list_dir (str): 生成的 train_list.txt, test_list.txt, label_list.txt 文件的存放目录。
        test_size (float): 测试集占总数据集的比例 (0.0 到 1.0 之间)。
        random_state (int): 随机种子，用于保证每次运行划分结果一致。
    """
    
    os.makedirs(output_list_dir, exist_ok=True) # 确保输出目录存在

    f_train = open(os.path.join(output_list_dir, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(output_list_dir, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(output_list_dir, 'label_list.txt'), 'w', encoding='utf-8')

    print(f"正在读取CSV元数据: {csv_filepath}")
    try:
        df = pd.read_csv(csv_filepath)
        df = df.dropna() # 确保没有NaN值
        print(f"成功读取 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误: CSV文件 '{csv_filepath}' 未找到。请检查路径。")
        return
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        return

    # 构建标签到ID的映射
    unique_labels = df['label'].unique()
    label_to_id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print(f"发现 {len(unique_labels)} 个唯一标签。")

    # 构建完整的音频路径和对应的类ID列表
    data_for_split = []
    missing_files_count = 0
    for index, row in df.iterrows():
        file_name = row['fname']
        label_name = row['label']
        class_id = label_to_id[label_name]

        sound_path = os.path.join(audio_base_dir, file_name).replace('\\', '/')

        if not os.path.exists(sound_path):
            missing_files_count += 1
            continue
        
        data_for_split.append({
            'sound_path': sound_path,
            'class_id': class_id,
            'label_name': label_name # 保留标签名称，用于分层抽样
        })
    
    if missing_files_count > 0:
        print(f"警告: 共有 {missing_files_count} 个音频文件在指定目录中不存在，已跳过。")

    if not data_for_split:
        print("没有有效的音频文件可以进行划分。")
        return

    # 将列表转换为DataFrame以便进行分层抽样划分
    data_df_for_split = pd.DataFrame(data_for_split)

    # 分层抽样划分训练集和测试集
    train_data, test_data = train_test_split(
        data_df_for_split,
        test_size=test_size,
        random_state=random_state,
        stratify=data_df_for_split['label_name'] # 关键：根据标签进行分层抽样
    )
    
    print(f"\n数据集划分完成: ")
    print(f"训练集: {len(train_data)} 条记录")
    print(f"测试集: {len(test_data)} 条记录")

    # 写入训练集文件
    for _, row in train_data.iterrows():
        f_train.write(f"{row['sound_path']}\t{row['class_id']}\n")

    # 写入测试集文件
    for _, row in test_data.iterrows():
        f_test.write(f"{row['sound_path']}\t{row['class_id']}\n")

    # 写入标签列表文件
    for i in range(len(id_to_label)):
        f_label.write(f'{id_to_label[i]}\n')

    f_label.close()
    f_test.close()
    f_train.close()
    print(f"\n列表文件已成功生成在目录: {output_list_dir}")
    print(f"训练集写入: {os.path.join(output_list_dir, 'train_list.txt')}")
    print(f"测试集写入: {os.path.join(output_list_dir, 'test_list.txt')}")
    print(f"标签列表写入: {os.path.join(output_list_dir, 'label_list.txt')}")
    print(f"总共处理并划分了 {len(data_df_for_split)} 个有效音频文件。")


# --- 使用示例 ---
if __name__ == "__main__":
    my_csv_file = "/home/renmengxing/audioRec/audio_data/train.csv"
    my_audio_dir = "/home/renmengxing/audioRec/audio_data/train"
    my_output_list_dir = "/home/user/gptdata/rmx/funaudio/AudioCls/code" # 建议使用新目录

    create_custom_audio_list_stratified_split(
        csv_filepath=my_csv_file,
        audio_base_dir=my_audio_dir,
        output_list_dir=my_output_list_dir,
        test_size=0.2, # 80% 训练，20% 测试
        random_state=10 # 保持一致的划分结果
    )