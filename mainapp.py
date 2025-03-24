## 依存ライブラリのインストール

##このコードを実行するには、以下のライブラリが必要です。

##```bash
##pip install janome jaconv

import os
import re
from janome.tokenizer import Tokenizer
import jaconv

# 設定
input_folder = './aozora_texts'
output_folder = './comparison_output'

def extract_hiragana(text):
    """
    テキストデータから可能な限りひらがなを抽出します（漢字は読みをひらがなで）。
    """
    tokenizer = Tokenizer() # udic パラメータを削除
    hiragana_words = []
    for token in tokenizer.tokenize(text):
        if token.reading != '*':
            # 読みが存在する場合は、カタカナをひらがなに変換して追加
            hiragana_words.append(jaconv.kata2hira(token.reading))
        elif all('\u3041' <= char <= '\u3096' for char in token.surface):
            # 読みがなく、表面形がひらがなの場合はそのまま追加
            hiragana_words.append(token.surface)
        elif token.part_of_speech.startswith('記号'):
            # 記号はそのまま追加
            hiragana_words.append(token.surface)
    return "".join(hiragana_words)

def preprocess_text(text):
    """
    テキストデータの前処理を行います（行ごと処理）。
    改行コードの削除、空白の正規化、およびカタカナへの変換を行います。
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    text = re.sub(r'\s+', ' ', text)

    tokenizer = Tokenizer() # udic パラメータを削除
    katakana_words = []
    for token in tokenizer.tokenize(text):
        if token.reading == '*':
            katakana_words.append(token.surface)
        else:
            katakana_words.append(token.reading)

    return "".join(katakana_words)

def read_text_with_bom_removal(filepath, encoding='utf-8'):
    """
    BOM付きの可能性のあるテキストファイルを読み込み、BOMを取り除いて返します。
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    if raw_data.startswith(b'\xef\xbb\xbf'):
        return raw_data[3:].decode(encoding)
    else:
        return raw_data.decode(encoding)

def output_comparison_data(filename, original_text, preprocessed_text, hiragana_text, output_folder):
    """
    元のテキスト、カタカナ変換後のテキスト、ひらがな抽出後のテキストを行ごとにタブ区切りでファイルに出力します。
    出力フォルダを引数として受け取ります。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_filename, ext = os.path.splitext(filename)
    output_filepath = os.path.join(output_folder, f"{base_filename}_comparison.tsv")

    try:
        with open(output_filepath, 'w', encoding='utf-8', errors='ignore') as outfile:
            outfile.write("Original\tKatakana\tHiragana\n")
            original_lines = original_text.splitlines()
            preprocessed_lines = preprocessed_text.splitlines()
            hiragana_lines = hiragana_text.splitlines()

            max_lines = max(len(original_lines), len(preprocessed_lines), len(hiragana_lines))
            for i in range(max_lines):
                original = original_lines[i] if i < len(original_lines) else ""
                preprocessed = preprocessed_lines[i] if i < len(preprocessed_lines) else ""
                hiragana = hiragana_lines[i] if i < len(hiragana_lines) else ""
                outfile.write(f"{original}\t{preprocessed}\t{hiragana}\n")

        print(f"比較データを '{output_filepath}' に出力しました。")

    except Exception as e:
        print(f"エラー: ファイル '{output_filepath}' への書き込みに失敗しました: {e}")

def process_file(filename, input_folder, output_folder):
    """
    指定されたファイル名で、input_folder内のテキストファイルを読み込み、
    行ごとに前処理（カタカナ変換）とひらがな抽出を行い、比較データを出力します。
    """
    filepath = os.path.join(input_folder, filename)
    try:
        original_text = read_text_with_bom_removal(filepath)
        original_lines = original_text.splitlines()
        preprocessed_lines = []
        hiragana_lines = []

        for line in original_lines:
            if line.strip():
                preprocessed_line = preprocess_text(line)
                hiragana_line = extract_hiragana(line)
                preprocessed_lines.append(preprocessed_line)
                hiragana_lines.append(hiragana_line)
            else:
                preprocessed_lines.append("") # 空行を保持
                hiragana_lines.append("")     # 空行を保持

        output_comparison_data(filename, original_text, "\n".join(preprocessed_lines), "\n".join(hiragana_lines), output_folder)

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"エラー: ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

def load_text_files(folder_path):
    """
    指定されたフォルダ内の .txt ファイルのリストを返します。
    """
    text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    return text_files

if __name__ == "__main__":
    try:
        import jaconv
    except ImportError:
        print("エラー: 'jaconv' ライブラリがインストールされていません。インストールするには 'pip install jaconv' を実行してください。")
        exit()

    text_files = load_text_files(input_folder)
    if not text_files:
        print(f"フォルダ '{input_folder}' にテキストファイルが見つかりませんでした。")
    else:
        for filename in text_files:
            print(f"処理中のファイル: {filename}")
            process_file(filename, input_folder, output_folder)
        print("すべてのファイルの処理が完了しました。")
