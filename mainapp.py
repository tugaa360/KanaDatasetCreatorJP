import os
import re
import argparse
import configparser
from janome.tokenizer import Tokenizer
import jaconv
from typing import List, Optional, Dict
import logging
import json
import tkinter as tk
from tkinter import filedialog, messagebox

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessingError(Exception):
    """テキスト処理に関するカスタム例外クラス"""
    pass

class FileProcessingError(TextProcessingError):
    """ファイル処理に関するカスタム例外クラス"""
    pass

class InvalidOutputFormatError(TextProcessingError):
    """無効な出力フォーマットに関するカスタム例外クラス"""
    pass

class TextProcessor:
    def __init__(self, udic_path: Optional[str] = None, remove_symbols: bool = True, convert_kanji: bool = False):
        """
        TextProcessorクラスの初期化

        Args:
            udic_path (Optional[str]): ユーザー辞書のパス。デフォルトはNone。
            remove_symbols (bool): 記号を削除するかどうか。デフォルトはTrue。
            convert_kanji (bool): 漢字をひらがなに変換するかどうか。デフォルトはFalse。
        """
        self.tokenizer = Tokenizer(udic=udic_path) if udic_path else Tokenizer()
        self.remove_symbols = remove_symbols
        self.convert_kanji = convert_kanji
        logger.info(f"TextProcessor initialized with udic_path: {udic_path}, remove_symbols: {remove_symbols}, convert_kanji: {convert_kanji}")

    def extract_hiragana(self, text: str) -> str:
        """
        テキストデータからひらがなを抽出します。

        Args:
            text (str): 処理するテキスト。

        Returns:
            str: ひらがな抽出後のテキスト。
        """
        hiragana_words: List[str] = []
        for token in self.tokenizer.tokenize(text):
            if token.reading != '*':
                # 読みが存在する場合は、ひらがなに変換
                reading = jaconv.h2z(token.reading, kana=True, ascii=False, digit=False)
                hiragana_words.append(jaconv.kata2hira(reading))
            elif all('\u3041' <= char <= '\u3096' for char in token.surface):
                # ひらがなの場合はそのまま追加
                hiragana_words.append(token.surface)
            elif all('\u4e00' <= char <= '\u9fff' for char in token.surface) and self.convert_kanji:
                # 漢字のみの場合は、変換を試みる
                # (ここに漢字をひらがなに変換するロジックを追加)
                # 今のところ、漢字はそのまま追加
                hiragana_words.append(token.surface)
            elif any('\u3041' <= char <= '\u3096' for char in token.surface) or any('\u4e00' <= char <= '\u9fff' for char in token.surface):
                # ひらがなと漢字が混ざっている場合は、可能な限りひらがな化
                mixed_word = ""
                for char in token.surface:
                    if '\u3041' <= char <= '\u3096':
                        mixed_word += char
                    elif '\u4e00' <= char <= '\u9fff' and self.convert_kanji:
                        mixed_word += char
                hiragana_words.append(mixed_word)
            elif token.part_of_speech.startswith('記号') and not self.remove_symbols:
                # 記号を削除しない設定の場合
                hiragana_words.append(token.surface)

        return "".join(hiragana_words)

    def preprocess_line(self, text: str, to_lower: bool = False, remove_punctuation: bool = False, remove_numbers: bool = False) -> str:
        """
        テキストデータの前処理を行います（行ごと処理）。

        Args:
            text (str): 処理するテキスト。
            to_lower (bool): 小文字に変換するかどうか。デフォルトはFalse。
            remove_punctuation (bool): 句読点を削除するかどうか。デフォルトはFalse。
            remove_numbers (bool): 数字を削除するかどうか。デフォルトはFalse。

        Returns:
            str: 前処理後のテキスト。
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
        text = re.sub(r'\s+', ' ', text)

        if to_lower:
            text = text.lower()

        if remove_punctuation:
            text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', text)

        if remove_numbers:
            text = re.sub(r'[0-9]', '', text)

        katakana_words = []
        for token in self.tokenizer.tokenize(text):
            if token.reading == '*':
                katakana_words.append(token.surface)
            else:
                katakana_words.append(token.reading)

        return "".join(katakana_words)

def read_text_with_bom_removal(filepath: str, encoding: str = 'utf-8') -> str:
    """
    BOM付きの可能性のあるテキストファイルを読み込み、BOMを取り除いて返します。

    Args:
        filepath (str): ファイルパス。
        encoding (str): ファイルのエンコーディング。デフォルトは'utf-8'。

    Returns:
        str: BOMを取り除いたテキストデータ。

    Raises:
        FileProcessingError: ファイルの読み込みに失敗した場合。
    """
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()

        if raw_data.startswith(b'\xef\xbb\xbf'):
            return raw_data[3:].decode(encoding)
        else:
            return raw_data.decode(encoding)
    except FileNotFoundError:
        raise FileProcessingError(f"ファイル '{filepath}' が見つかりません。")
    except UnicodeDecodeError as e:
        raise FileProcessingError(f"ファイル '{filepath}' のデコードに失敗しました: {e}")
    except Exception as e:
        raise FileProcessingError(f"ファイル '{filepath}' の読み込み中に予期しないエラーが発生しました: {e}")

def output_comparison_data(filename: str, original_text: str, preprocessed_text: str, hiragana_text: str, output_folder: str, output_format: str = 'tsv') -> None:
    """
    元のテキスト、カタカナ変換後のテキスト、ひらがな抽出後のテキストを行ごとに指定されたフォーマットでファイルに出力します。
    TSVとJSONLの両方を出力するように変更。
    """
    if not os.path.exists(output_folder):
        # ディレクトリがない場合、作成するか確認
        if messagebox.askyesno("ディレクトリ作成", f"出力ディレクトリ '{output_folder}' が存在しません。作成しますか？"):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                raise FileProcessingError(f"出力ディレクトリ '{output_folder}' の作成に失敗しました: {e}")
        else:
            raise FileProcessingError(f"出力ディレクトリ '{output_folder}' が存在しないため、処理を中止します。")

    base_filename, ext = os.path.splitext(filename)
    output_tsv_filepath = os.path.join(output_folder, f"{base_filename}_comparison.tsv")
    output_jsonl_filepath = os.path.join(output_folder, f"{base_filename}_comparison.jsonl")

    try:
        with open(output_tsv_filepath, 'w', encoding='utf-8', errors='replace') as tsvfile, \
             open(output_jsonl_filepath, 'w', encoding='utf-8', errors='replace') as jsonlfile:

            tsvfile.write("Original\tKatakana\tHiragana\n")
            original_lines = original_text.splitlines()
            preprocessed_lines = preprocessed_text.splitlines()
            hiragana_lines = hiragana_text.splitlines()

            max_lines = max(len(original_lines), len(preprocessed_lines), len(hiragana_lines))
            for i in range(max_lines):
                original = original_lines[i] if i < len(original_lines) else ""
                preprocessed = preprocessed_lines[i] if i < len(preprocessed_lines) else ""
                hiragana = hiragana_lines[i] if i < len(hiragana_lines) else ""

                # TSVファイルへの書き込み
                tsvfile.write(f"{original}\t{preprocessed}\t{hiragana}\n")

                # JSONLファイルへの書き込み
                json_data = {
                    "Original": original,
                    "Katakana": preprocessed,
                    "Hiragana": hiragana
                }
                jsonlfile.write(json.dumps(json_data, ensure_ascii=False) + '\n')

        logger.info(f"比較データを '{output_tsv_filepath}' に出力しました。")
        logger.info(f"比較データを '{output_jsonl_filepath}' に出力しました。")

    except Exception as e:
        raise FileProcessingError(f"ファイルへの書き込みに失敗しました: {e}")

def process_file(filepath: str, output_folder: str, text_processor: TextProcessor, output_format: str) -> None:
    """
    指定されたテキストファイルを読み込み、行ごとに前処理（カタカナ変換）とひらがな抽出を行い、比較データを出力します。

    Args:
        filepath (str): 処理するファイルのパス。
        output_folder (str): 出力フォルダのパス。
        text_processor (TextProcessor): テキスト処理を行うTextProcessorオブジェクト。
        output_format (str): 出力フォーマット（'tsv'または'csv'）。

    Raises:
        FileProcessingError: ファイルの読み込みまたは処理中にエラーが発生した場合。
    """
    filename = os.path.basename(filepath)
    try:
        original_text = read_text_with_bom_removal(filepath)
        original_lines = original_text.splitlines()
        preprocessed_lines: List[str] = []
        hiragana_lines: List[str] = []

        for line in original_lines:
            if line.strip():
                preprocessed_line = text_processor.preprocess_line(line)
                hiragana_line = text_processor.extract_hiragana(line)
                preprocessed_lines.append(preprocessed_line)
                hiragana_lines.append(hiragana_line)
            else:
                preprocessed_lines.append("")  # 空行を保持
                hiragana_lines.append("")  # 空行を保持

        output_comparison_data(filename, original_text, "\n".join(preprocessed_lines), "\n".join(hiragana_lines), output_folder, output_format)

    except FileProcessingError as e:
        raise e
    except Exception as e:
        raise FileProcessingError(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

def select_files():
    """ファイルを選択するダイアログを表示します。"""
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示
    file_paths = filedialog.askopenfilenames(title="処理するテキストファイルを選択してください", filetypes=[("Text files", "*.txt")])
    return list(file_paths)

def select_output_folder():
    """出力フォルダを選択するダイアログを表示します。"""
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示
    folder_path = filedialog.askdirectory(title="出力フォルダを選択してください")
    return folder_path

def main():
    """
    メイン関数。テキストファイルの前処理とひらがな抽出を行います。
    """
    parser = argparse.ArgumentParser(description='テキストファイルの前処理とひらがな抽出を行います。')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--udic_path', type=str, help='udicのパス')
    parser.add_argument('--file_extension', type=str, default='.txt', help='処理するファイルの拡張子')
    parser.add_argument('--output_format', type=str, default='tsv', choices=['tsv', 'csv'], help='出力フォーマット（tsvまたはcsv）')
    parser.add_argument('--remove_symbols', action='store_true', help='記号を削除する')
    parser.add_argument('--convert_kanji', action='store_true', help='漢字をひらがなに変換する')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if args.config and os.path.exists(args.config):
        config.read(args.config)
        output_folder = config.get('Paths', 'output_folder', fallback=None)
        udic_path = config.get('Paths', 'udic_path', fallback=args.udic_path)
        file_extension = config.get('Settings', 'file_extension', fallback=args.file_extension)
        output_format = config.get('Settings', 'output_format', fallback=args.output_format)
        remove_symbols = config.getboolean('Settings', 'remove_symbols', fallback=not args.remove_symbols)
        convert_kanji = config.getboolean('Settings', 'convert_kanji', fallback=args.convert_kanji)
    else:
        output_folder = None
        udic_path = args.udic_path
        file_extension = args.file_extension
        output_format = args.output_format
        remove_symbols = not args.remove_symbols
        convert_kanji = args.convert_kanji

    # ファイル選択ダイアログを表示
    file_paths = select_files()
    if not file_paths:
        logger.info("ファイルが選択されなかったため、処理を中止します。")
        return

    # 出力フォルダ選択ダイアログを表示
    if output_folder is None:
        output_folder = select_output_folder()
        if not output_folder:
            logger.info("出力フォルダが選択されなかったため、処理を中止します。")
            return

    # パスの正規化
    output_folder = os.path.normpath(output_folder)

    text_processor = TextProcessor(udic_path, remove_symbols, convert_kanji)

    try:
        total_files = len(file_paths)
        for i, filepath in enumerate(file_paths):
            logger.info(f"処理中のファイル ({i+1}/{total_files}): {filepath}")
            try:
                process_file(filepath, output_folder, text_processor, output_format)
            except FileProcessingError as e:
                logger.error(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")
        logger.info("すべてのファイルの処理が完了しました。")
    except FileProcessingError as e:
        logger.error(f"ファイルの処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
