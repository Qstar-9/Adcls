import pandas as pd
import requests
import os
import json
import time

api_url = "http://202.127.200.34:30014/chat"
test_csv_path = "/home/renmengxing/audioRec/test.csv"
output_results_csv = "/home/renmengxing/audioRec/test_predictions.csv"

audio_categories_dict = {
    'Hi-hat': 300, 'Laughter': 300, 'Shatter': 300, 'Applause': 300, 'Squeak': 300,
    'Acoustic_guitar': 300, 'Bass_drum': 300, 'Saxophone': 300, 'Flute': 300,
    'Double_bass': 300, 'Tearing': 300, 'Fart': 300, 'Clarinet': 300,
    'Fireworks': 300, 'Trumpet': 300, 'Violin_or_fiddle': 300, 'Cello': 300,
    'Snare_drum': 300, 'Oboe': 299, 'Gong': 292, 'Knock': 279, 'Writing': 270,
    'Cough': 243, 'Bark': 239, 'Tambourine': 221, 'Burping_or_eructation': 210,
    'Cowbell': 191, 'Harmonica': 165, 'Drawer_open_or_close': 158, 'Meow': 155,
    'Electric_piano': 150, 'Gunshot_or_gunfire': 147, 'Microwave_oven': 146,
    'Keys_jangling': 139, 'Telephone': 120, 'Computer_keyboard': 119,
    'Finger_snapping': 117, 'Chime': 115, 'Bus': 109, 'Scissors': 95,
    'Glockenspiel': 94
}

audio_category_names = list(audio_categories_dict.keys())
audio_categories_str = ", ".join(audio_category_names)

fixed_text_input = f"""
Please output the top three most probable results, sorted from highest to lowest confidence, separated by English commas. For example: `Hi-hat, Snare_drum, Bass_drum`.
Based on the input audio content, identify and determine which sound category the audio most likely belongs to.
You must choose from the following 41 audio categories only. Do not generate any other categories:
{audio_categories_str}
"""

def run_inference_on_test_set(api_endpoint: str, test_data_csv: str, output_csv: str, text_prompt: str):
    try:
        test_df = pd.read_csv(test_data_csv)
        print(f"成功读取测试集文件: {test_data_csv}, 共有 {len(test_df)} 条记录。")
    except FileNotFoundError:
        print(f"错误: 测试集文件 '{test_data_csv}' 未找到。请检查路径。")
        return
    except Exception as e:
        print(f"读取测试集文件时发生错误: {e}")
        return

    predictions = []
    
    try:
        from tqdm import tqdm
        iter_items = tqdm(test_df.iterrows(), total=len(test_df), desc="推理进度")
    except ImportError:
        iter_items = test_df.iterrows()

    for index, row in iter_items:
        file_name = row['fname']
        audio_full_path = row['absolute_audio_path']

        if not os.path.exists(audio_full_path):
            print(f"警告: 音频文件 '{audio_full_path}' 不存在，跳过此文件。")
            predictions.append({
                'fname': file_name,
                'model_response': f"ERROR: File not found: {audio_full_path}",
                'predicted_labels': ""
            })
            continue

        try:
            with open(audio_full_path, 'rb') as f:
                files = {'audio_file': (file_name, f, 'audio/wav')}
                data = {'text_input': text_prompt} 

                response = requests.post(api_endpoint, files=files, data=data)
                response.raise_for_status()

                result = response.json()
                model_response_text = result.get('response', '').strip()

                predicted_labels_for_submission = ""
                if model_response_text:
                    parsed_labels = [label.strip() for label in model_response_text.split(',') if label.strip()]
                    predicted_labels_for_submission = " ".join(parsed_labels[:3])

                predictions.append({
                    'fname': file_name,
                    'model_response': model_response_text,
                    'predicted_labels': predicted_labels_for_submission
                })

        except requests.exceptions.RequestException as req_err:
            print(f"请求错误 (文件: {file_name}): {req_err}")
            predictions.append({
                'fname': file_name,
                'model_response': f"ERROR: Request failed - {req_err}",
                'predicted_labels': ""
            })
        except json.JSONDecodeError:
            print(f"JSON 解析错误 (文件: {file_name}): 响应内容不是有效的 JSON。")
            predictions.append({
                'fname': file_name,
                'model_response': f"ERROR: Invalid JSON response",
                'predicted_labels': ""
            })
        except Exception as e:
            print(f"处理文件 '{file_name}' 时发生未知错误: {e}")
            predictions.append({
                'fname': file_name,
                'model_response': f"ERROR: Unexpected error - {e}",
                'predicted_labels': ""
            })

    results_df = pd.DataFrame(predictions)
    
    results_df.to_csv(output_csv, index=False)
    print(f"\n所有推理结果已保存到: {output_csv}")
    print("\n结果文件前5行预览:")
    print(results_df.head())

if __name__ == "__main__":
    run_inference_on_test_set(api_url, test_csv_path, output_results_csv, fixed_text_input)