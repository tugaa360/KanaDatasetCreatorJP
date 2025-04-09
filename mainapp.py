import os
import re
import argparse
import configparser
from janome.tokenizer import Tokenizer
import jaconv
from typing import List, Optional, Dict, Tuple
import logging
import json
import sys

# --- オプション: GUI機能の有効化 ---
# tkinterがインストールされていない、またはGUI不要な場合はFalseにする
ENABLE_GUI = True
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:
    ENABLE_GUI = False
    logging.warning("tkinterが見つからないため、GUI機能（ファイル/フォルダ選択ダイアログ、確認ダイアログ）は無効になります。")

# --- オプション: 漢字->ひらがな変換機能の有効化 ---
# pykakasiがインストールされていない場合はFalseにする
ENABLE_KANJI_CONVERSION = True
try:
    # pip install pykakasi
    from pykakasi import kakasi
except ImportError:
    ENABLE_KANJI_CONVERSION = False
    logging.warning("pykakasiが見つからないため、漢字からひらがなへの変換機能は無効になります (`convert_kanji` オプションは無視されます)。")

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessingError(Exception):
    """テキスト処理に関するカスタム例外クラス"""
    pass

class FileProcessingError(TextProcessingError):
    """ファイル処理に関するカスタム例外クラス"""
    pass

class InitializationError(TextProcessingError):
    """初期化に関するカスタム例外クラス"""
    pass

class TextProcessor:
    def __init__(self, udic_path: Optional[str] = None, remove_symbols: bool = True, convert_kanji: bool = False):
        """
        TextProcessorクラスの初期化

        Args:
            udic_path (Optional[str]): ユーザー辞書のパス。デフォルトはNone。
            remove_symbols (bool): 記号を削除するかどうか。デフォルトはTrue。
            convert_kanji (bool): 漢字をひらがなに変換するかどうか。デフォルトはFalse。
                                   (pykakasiがインストールされている場合のみ有効)

        Raises:
            InitializationError: Tokenizerの初期化に失敗した場合。
            InitializationError: convert_kanji=Trueだがpykakasiが利用できない場合。
        """
        try:
            self.tokenizer = Tokenizer(udic=udic_path, udic_enc='utf8') if udic_path else Tokenizer()
        except Exception as e:
            raise InitializationError(f"Tokenizerの初期化に失敗しました (udic_path: {udic_path}): {e}")

        self.remove_symbols = remove_symbols
        self.convert_kanji = convert_kanji and ENABLE_KANJI_CONVERSION

        if convert_kanji and not ENABLE_KANJI_CONVERSION:
             logger.warning("`convert_kanji=True` が指定されましたが、pykakasiが利用できないため無効化されます。")
             self.convert_kanji = False # 強制的にFalseにする

        self.kks = None
        self.conv = None
        if self.convert_kanji:
            try:
                self.kks = kakasi()
                # 必要に応じて変換モードを設定 (例: ひらがな出力)
                self.kks.setMode("J", "H") # Kanji to Hiragana
                self.kks.setMode("K", "H") # Katakana to Hiragana
                self.kks.setMode("H", "H") # Hiragana to Hiragana (そのまま)
                self.kks.setMode("E", "H") # Alphabet to Hiragana (必要なら)
                self.kks.setMode("r", "H") # Romaji to Hiragana (必要なら)
                self.conv = self.kks.getConverter()
                logger.info("漢字->ひらがな変換機能 (pykakasi) を有効化しました。")
            except Exception as e:
                raise InitializationError(f"pykakasiの初期化に失敗しました: {e}")

        logger.info(f"TextProcessor initialized with: udic_path='{udic_path}', remove_symbols={remove_symbols}, convert_kanji={self.convert_kanji}")

    def _safe_convert_to_hiragana(self, text: str) -> Optional[str]:
        """pykakasiで安全にひらがな変換を試みる内部メソッド"""
        if not self.conv:
            return None
        try:
            converted = self.conv.do(text)
            # 変換後がひらがなかどうかをチェック (より厳密に)
            if all('\u3040' <= char <= '\u309F' for char in converted): # ひらがな、長音符、繰り返し記号
                return converted
            else:
                logger.debug(f"Skipping non-hiragana conversion result for '{text}': {converted}")
                return None # ひらがな以外が含まれていたらNoneを返す
        except Exception as e:
            logger.warning(f"Kanji to Hiragana conversion failed for '{text}': {e}")
            return None

    def extract_hiragana(self, text: str) -> str:
        """
        テキストデータからひらがなを抽出します。漢字は設定に応じてひらがなに変換します。

        Args:
            text (str): 処理するテキスト。

        Returns:
            str: ひらがな抽出・変換後のテキスト。
        """
        hiragana_parts: List[str] = []
        cleaned_text = text.replace('\r\n', '').replace('\r', '').replace('\n', '') # 改行は除去してから処理

        for token in self.tokenizer.tokenize(cleaned_text):
            part_of_speech = token.part_of_speech.split(',')[0]

            # 1. reading が利用可能なら優先 (カタカナ -> ひらがな)
            if token.reading != '*':
                hiragana_parts.append(jaconv.kata2hira(token.reading))
                continue # 次のトークンへ

            # 2. reading がなく、表層形がひらがなのみ
            #   (厳密には長音符なども考慮するべきだが、ここではひらがな文字コード範囲で判定)
            is_hiragana_only = all('\u3041' <= char <= '\u3096' for char in token.surface)
            if is_hiragana_only:
                hiragana_parts.append(token.surface)
                continue

            # 3. reading がなく、漢字変換が有効で、変換を試みる
            if self.convert_kanji:
                converted_hira = self._safe_convert_to_hiragana(token.surface)
                if converted_hira is not None:
                    hiragana_parts.append(converted_hira)
                    continue

            # 4. 記号で、かつ記号を保持する場合
            if part_of_speech == '記号' and not self.remove_symbols:
                hiragana_parts.append(token.surface)
                continue

            # 5. 上記以外（記号削除時、アルファベット、数字、変換不可の漢字など）は無視
            logger.debug(f"Ignoring token: surface='{token.surface}', reading='{token.reading}', pos='{token.part_of_speech}'")

        return "".join(hiragana_parts)

    def preprocess_line(self, text: str) -> str:
        """
        テキストデータの前処理を行います（行ごと処理、カタカナ変換）。
        句読点や数字の削除はここでは行わない（必要なら別途実装）。

        Args:
            text (str): 処理するテキスト。

        Returns:
            str: 前処理後のテキスト（カタカナ）。
        """
        # 改行を除去し、前後の空白を削除
        text = text.replace('\r\n', '').replace('\r', '').replace('\n', '').strip()
        # 連続する空白を一つにまとめる (ここでは不要かもしれないが念のため)
        # text = re.sub(r'\s+', ' ', text)

        katakana_words = []
        for token in self.tokenizer.tokenize(text):
            if token.reading == '*':
                # readingがない場合は表層形をそのまま使う
                katakana_words.append(token.surface)
            else:
                # readingがある場合はそれを使う (通常カタカナ)
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

        # BOM check
        if raw_data.startswith(b'\xef\xbb\xbf'): # UTF-8 BOM
            return raw_data[3:].decode(encoding, errors='replace')
        elif raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'): # UTF-16 BOM (LE/BE)
             # UTF-16の可能性が高い場合、encoding引数を無視して推測する方が安全かも
            try:
                return raw_data.decode('utf-16', errors='replace')
            except UnicodeDecodeError:
                 logger.warning(f"Detected potential UTF-16 BOM in '{filepath}', but decoding failed. Falling back to '{encoding}'.")
                 return raw_data.decode(encoding, errors='replace') # フォールバック
        else:
            # BOMがない場合は指定されたエンコーディングでデコード
            return raw_data.decode(encoding, errors='replace')
    except FileNotFoundError:
        raise FileProcessingError(f"ファイル '{filepath}' が見つかりません。")
    except UnicodeDecodeError as e:
        raise FileProcessingError(f"ファイル '{filepath}' のデコードに失敗しました (encoding: {encoding}): {e}")
    except Exception as e:
        raise FileProcessingError(f"ファイル '{filepath}' の読み込み中に予期しないエラーが発生しました: {e}")

def output_comparison_data(
    filename: str,
    original_text: str,
    preprocessed_text: str,
    hiragana_text: str,
    output_folder: str,
    output_formats: List[str]
) -> None:
    """
    元のテキスト、カタカナ変換後のテキスト、ひらがな抽出後のテキストを行ごとに指定されたフォーマットでファイルに出力します。

    Args:
        filename (str): 元のファイル名。
        original_text (str): 元のテキストデータ（複数行含む）。
        preprocessed_text (str): カタカナ変換後のテキストデータ（複数行含む）。
        hiragana_text (str): ひらがな抽出後のテキストデータ（複数行含む）。
        output_folder (str): 出力フォルダのパス。
        output_formats (List[str]): 出力するフォーマットのリスト（例: ['tsv', 'jsonl']）。

    Raises:
        FileProcessingError: ディレクトリ作成またはファイル書き込みに失敗した場合。
    """
    if not os.path.exists(output_folder):
        logger.warning(f"出力ディレクトリ '{output_folder}' が存在しません。作成を試みます。")
        # GUIが有効で、かつユーザー確認がTrueの場合のみ作成、それ以外は警告ログのみで作成試行
        create_dir = True
        if ENABLE_GUI:
            try:
                # GUI環境下でのみ確認ダイアログを表示
                if not messagebox.askyesno("ディレクトリ作成確認", f"出力ディレクトリ '{output_folder}' が存在しません。作成しますか？"):
                    create_dir = False
            except Exception as e:
                logger.warning(f"GUI確認ダイアログの表示に失敗しました: {e}. ディレクトリ作成を試みます。")

        if create_dir:
            try:
                os.makedirs(output_folder)
                logger.info(f"出力ディレクトリ '{output_folder}' を作成しました。")
            except Exception as e:
                raise FileProcessingError(f"出力ディレクトリ '{output_folder}' の作成に失敗しました: {e}")
        else:
            raise FileProcessingError(f"出力ディレクトリ '{output_folder}' が存在せず、作成がキャンセルされたため処理を中止します。")


    base_filename, _ = os.path.splitext(filename)
    output_basepath = os.path.join(output_folder, f"{base_filename}_comparison")

    original_lines = original_text.splitlines()
    preprocessed_lines = preprocessed_text.splitlines()
    hiragana_lines = hiragana_text.splitlines()
    # 各行数が異なる場合も考慮し、最も長い行数に合わせる
    max_lines = max(len(original_lines), len(preprocessed_lines), len(hiragana_lines))

    output_successful = []

    # --- TSV Output ---
    if 'tsv' in output_formats:
        output_tsv_filepath = f"{output_basepath}.tsv"
        try:
            with open(output_tsv_filepath, 'w', encoding='utf-8', errors='replace', newline='') as tsvfile:
                # ヘッダー書き込み
                tsvfile.write("Original\tKatakana\tHiragana\n")
                # データ書き込み
                for i in range(max_lines):
                    original = original_lines[i] if i < len(original_lines) else ""
                    preprocessed = preprocessed_lines[i] if i < len(preprocessed_lines) else ""
                    hiragana = hiragana_lines[i] if i < len(hiragana_lines) else ""
                    # タブ文字や改行文字が含まれているとTSVが壊れるため、置換するかエスケープが必要
                    # ここでは単純にスペースに置換する例 (必要に応じて調整)
                    original_safe = original.replace('\t', ' ').replace('\n', ' ').replace('\r', '')
                    preprocessed_safe = preprocessed.replace('\t', ' ').replace('\n', ' ').replace('\r', '')
                    hiragana_safe = hiragana.replace('\t', ' ').replace('\n', ' ').replace('\r', '')
                    tsvfile.write(f"{original_safe}\t{preprocessed_safe}\t{hiragana_safe}\n")
            logger.info(f"比較データをTSV形式で '{output_tsv_filepath}' に出力しました。")
            output_successful.append('tsv')
        except Exception as e:
            logger.error(f"TSVファイル '{output_tsv_filepath}' への書き込みに失敗しました: {e}")

    # --- JSONL Output ---
    if 'jsonl' in output_formats:
        output_jsonl_filepath = f"{output_basepath}.jsonl"
        try:
            with open(output_jsonl_filepath, 'w', encoding='utf-8', errors='replace') as jsonlfile:
                for i in range(max_lines):
                    original = original_lines[i] if i < len(original_lines) else ""
                    preprocessed = preprocessed_lines[i] if i < len(preprocessed_lines) else ""
                    hiragana = hiragana_lines[i] if i < len(hiragana_lines) else ""
                    json_data = {
                        "Original": original,
                        "Katakana": preprocessed,
                        "Hiragana": hiragana
                    }
                    # ensure_ascii=False で日本語をそのまま出力
                    jsonlfile.write(json.dumps(json_data, ensure_ascii=False) + '\n')
            logger.info(f"比較データをJSONL形式で '{output_jsonl_filepath}' に出力しました。")
            output_successful.append('jsonl')
        except Exception as e:
            logger.error(f"JSONLファイル '{output_jsonl_filepath}' への書き込みに失敗しました: {e}")

    # --- CSV Output (Placeholder) ---
    if 'csv' in output_formats:
        output_csv_filepath = f"{output_basepath}.csv"
        logger.warning(f"CSV形式の出力は現在サポートされていません。'{output_csv_filepath}' は作成されません。")
        # 必要であればここにCSV出力ロジックを追加 (csvモジュール推奨)
        # import csv
        # try:
        #     with open(output_csv_filepath, 'w', encoding='utf-8', errors='replace', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow(["Original", "Katakana", "Hiragana"])
        #         for i in range(max_lines):
        #             # ... (データ取得、必要ならエスケープ処理)
        #             writer.writerow([original, preprocessed, hiragana])
        #     logger.info(f"比較データをCSV形式で '{output_csv_filepath}' に出力しました。")
        #     output_successful.append('csv')
        # except Exception as e:
        #      logger.error(f"CSVファイル '{output_csv_filepath}' への書き込みに失敗しました: {e}")

    if not output_successful:
        raise FileProcessingError(f"ファイル '{filename}' のいずれの指定されたフォーマットでも出力に成功しませんでした。")


def process_file(filepath: str, output_folder: str, text_processor: TextProcessor, output_formats: List[str]) -> None:
    """
    指定されたテキストファイルを読み込み、行ごとに前処理（カタカナ変換）とひらがな抽出を行い、比較データを出力します。

    Args:
        filepath (str): 処理するファイルのパス。
        output_folder (str): 出力フォルダのパス。
        text_processor (TextProcessor): テキスト処理を行うTextProcessorオブジェクト。
        output_formats (List[str]): 出力フォーマットのリスト。

    Raises:
        FileProcessingError: ファイルの読み込みまたは処理中にエラーが発生した場合。
    """
    filename = os.path.basename(filepath)
    try:
        logger.info(f"ファイル '{filename}' の読み込みを開始します...")
        original_text = read_text_with_bom_removal(filepath)
        original_lines = original_text.splitlines()
        logger.info(f"ファイル '{filename}' の読み込み完了。行数: {len(original_lines)}")

        preprocessed_lines: List[str] = []
        hiragana_lines: List[str] = []

        logger.info(f"ファイル '{filename}' の処理を開始します...")
        for i, line in enumerate(original_lines):
            # 空行や空白のみの行はそのまま（または空文字として）保持
            if line.strip():
                try:
                    preprocessed_line = text_processor.preprocess_line(line)
                    hiragana_line = text_processor.extract_hiragana(line)
                    preprocessed_lines.append(preprocessed_line)
                    hiragana_lines.append(hiragana_line)
                except Exception as e:
                    logger.error(f"ファイル '{filename}' の {i+1}行目 '{line[:50]}...' の処理中にエラーが発生しました: {e}")
                    # エラーが発生した行は空文字として扱うか、スキップするか選択
                    preprocessed_lines.append(f"ERROR: {e}") # エラー情報を記録
                    hiragana_lines.append(f"ERROR: {e}")
            else:
                preprocessed_lines.append("")  # 空行を保持
                hiragana_lines.append("")  # 空行を保持
        logger.info(f"ファイル '{filename}' の処理完了。")

        logger.info(f"ファイル '{filename}' の比較データの出力を開始します...")
        # 処理後のテキストを結合
        preprocessed_full_text = "\n".join(preprocessed_lines)
        hiragana_full_text = "\n".join(hiragana_lines)

        output_comparison_data(
            filename,
            original_text,
            preprocessed_full_text,
            hiragana_full_text,
            output_folder,
            output_formats
        )
        logger.info(f"ファイル '{filename}' の比較データの出力完了。")

    except FileProcessingError as e:
        # FileProcessingErrorは既にログされている可能性があるので再raise
        raise e
    except Exception as e:
        # 予期しないエラー
        raise FileProcessingError(f"ファイル '{filepath}' の処理中に予期しないエラーが発生しました: {e}")


def select_files_gui(file_extension: str = ".txt") -> List[str]:
    """ファイルを選択するGUIダイアログを表示します。"""
    if not ENABLE_GUI:
        logger.error("GUIが無効なため、ファイル選択ダイアログを表示できません。--input_files 引数を使用してください。")
        return []
    try:
        root = tk.Tk()
        root.withdraw()  # メインウィンドウを非表示
        filetypes = [(f"テキストファイル (*{file_extension})", f"*{file_extension}"), ("すべてのファイル", "*.*")]
        file_paths = filedialog.askopenfilenames(title=f"処理するファイルを選択してください ({file_extension})", filetypes=filetypes)
        root.destroy() # ダイアログが閉じたらTkインスタンスを破棄
        return list(file_paths)
    except Exception as e:
        logger.error(f"ファイル選択ダイアログの表示中にエラーが発生しました: {e}")
        return []

def select_output_folder_gui() -> Optional[str]:
    """出力フォルダを選択するGUIダイアログを表示します。"""
    if not ENABLE_GUI:
        logger.error("GUIが無効なため、フォルダ選択ダイアログを表示できません。--output_folder 引数を使用してください。")
        return None
    try:
        root = tk.Tk()
        root.withdraw()  # メインウィンドウを非表示
        folder_path = filedialog.askdirectory(title="出力フォルダを選択してください")
        root.destroy() # ダイアログが閉じたらTkインスタンスを破棄
        return folder_path if folder_path else None # キャンセル時は空文字列が返るためNoneに変換
    except Exception as e:
        logger.error(f"フォルダ選択ダイアログの表示中にエラーが発生しました: {e}")
        return None

def main():
    """
    メイン関数。テキストファイルの前処理とひらがな抽出を行います。
    """
    parser = argparse.ArgumentParser(description='日本語テキストファイルの前処理（カタカナ変換）とひらがな抽出を行い、比較データを出力します。')
    parser.add_argument('--config', type=str, help='設定ファイル (.ini) のパス')
    parser.add_argument('--input_files', type=str, nargs='+', help='処理する入力ファイルのパス（複数指定可）。指定しない場合はGUIで選択（GUI有効時）。')
    parser.add_argument('--output_folder', type=str, help='出力フォルダのパス。指定しない場合はGUIで選択（GUI有効時）。')
    parser.add_argument('--udic_path', type=str, help='Janome用ユーザー辞書 (.csv) のパス')
    parser.add_argument('--file_extension', type=str, default='.txt', help='処理対象とするファイルの拡張子 (GUI選択時やディレクトリ指定時に使用)')
    parser.add_argument('--output_format', type=str, nargs='+', default=['tsv', 'jsonl'], choices=['tsv', 'csv', 'jsonl'], help='出力フォーマット（複数選択可: tsv, csv, jsonl）')
    parser.add_argument('--keep_symbols', action='store_true', help='記号を削除せずに保持する (デフォルト: 削除)')
    parser.add_argument('--convert_kanji', action='store_true', help=f'漢字をひらがなに変換する (pykakasi が必要, 利用可能か: {ENABLE_KANJI_CONVERSION})')
    parser.add_argument('--encoding', type=str, default='utf-8', help='入力ファイルのエンコーディング')

    args = parser.parse_args()

    # --- 設定の読み込み ---
    config = configparser.ConfigParser()
    settings = {
        'input_files': None,
        'output_folder': None,
        'udic_path': None,
        'file_extension': args.file_extension,
        'output_format': args.output_format,
        'remove_symbols': not args.keep_symbols, # 引数と逆
        'convert_kanji': args.convert_kanji,
        'encoding': args.encoding,
    }

    if args.config and os.path.exists(args.config):
        try:
            config.read(args.config, encoding='utf-8')
            logger.info(f"設定ファイル '{args.config}' を読み込みました。")

            # コマンドライン引数より設定ファイルを優先する項目
            settings['output_folder'] = config.get('Paths', 'output_folder', fallback=settings['output_folder'])
            settings['udic_path'] = config.get('Paths', 'udic_path', fallback=settings['udic_path'])
            # input_files は設定ファイルでは指定しない想定 (またはディレクトリ指定などの拡張が必要)

            # コマンドライン引数で指定されていない場合のみ設定ファイルを適用する項目
            if args.file_extension == parser.get_default('file_extension'):
                 settings['file_extension'] = config.get('Settings', 'file_extension', fallback=settings['file_extension'])
            if args.output_format == parser.get_default('output_format'):
                output_format_str = config.get('Settings', 'output_format', fallback=None)
                if output_format_str:
                    # カンマ区切りなどで複数のフォーマットを指定可能にする
                    settings['output_format'] = [fmt.strip() for fmt in output_format_str.split(',') if fmt.strip() in ['tsv', 'csv', 'jsonl']]
            # keep_symbols / remove_symbols
            if not args.keep_symbols: # コマンドラインで --keep_symbols が指定されていない場合
                 # 設定ファイルで remove_symbols を見て、その逆を keep_symbols とする
                 remove_symbols_conf = config.getboolean('Settings', 'remove_symbols', fallback=True) # デフォルトは削除
                 settings['remove_symbols'] = remove_symbols_conf
            # convert_kanji
            if not args.convert_kanji: # コマンドラインで --convert_kanji が指定されていない場合
                 settings['convert_kanji'] = config.getboolean('Settings', 'convert_kanji', fallback=False)
            # encoding
            if args.encoding == parser.get_default('encoding'):
                settings['encoding'] = config.get('Settings', 'encoding', fallback=settings['encoding'])

        except configparser.Error as e:
            logger.warning(f"設定ファイル '{args.config}' の読み込みに失敗しました: {e}")
        except Exception as e:
             logger.warning(f"設定ファイル処理中に予期しないエラー: {e}")

    # コマンドライン引数が指定されていれば最優先
    if args.input_files:
        settings['input_files'] = args.input_files
    if args.output_folder:
        settings['output_folder'] = args.output_folder
    if args.udic_path:
        settings['udic_path'] = args.udic_path
    # remove_symbols は args.keep_symbols から決定
    settings['remove_symbols'] = not args.keep_symbols
    # convert_kanji は args.convert_kanji を優先
    settings['convert_kanji'] = args.convert_kanji
    # output_format は args.output_format を優先
    settings['output_format'] = args.output_format
    # encoding は args.encoding を優先
    settings['encoding'] = args.encoding


    # --- 入力ファイルの決定 ---
    file_paths = settings['input_files']
    if not file_paths:
        if ENABLE_GUI:
            logger.info("入力ファイルが指定されていないため、GUIで選択します。")
            file_paths = select_files_gui(settings['file_extension'])
            if not file_paths:
                logger.info("ファイルが選択されなかったため、処理を中止します。")
                sys.exit(0)
        else:
            logger.error("入力ファイルが指定されていません。--input_files 引数で指定してください。")
            sys.exit(1)

    # --- 出力フォルダの決定 ---
    output_folder = settings['output_folder']
    if not output_folder:
        if ENABLE_GUI:
            logger.info("出力フォルダが指定されていないため、GUIで選択します。")
            output_folder = select_output_folder_gui()
            if not output_folder:
                logger.info("出力フォルダが選択されなかったため、処理を中止します。")
                sys.exit(0)
        else:
            logger.error("出力フォルダが指定されていません。--output_folder 引数で指定してください。")
            sys.exit(1)

    # パスの正規化
    output_folder = os.path.normpath(output_folder)
    udic_path = os.path.normpath(settings['udic_path']) if settings['udic_path'] else None

    # --- TextProcessorの初期化 ---
    try:
        text_processor = TextProcessor(
            udic_path=udic_path,
            remove_symbols=settings['remove_symbols'],
            convert_kanji=settings['convert_kanji']
        )
    except InitializationError as e:
        logger.error(f"TextProcessorの初期化に失敗しました: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"TextProcessorの初期化中に予期しないエラーが発生しました: {e}")
        sys.exit(1)


    # --- ファイル処理の実行 ---
    logger.info(f"処理を開始します。入力ファイル数: {len(file_paths)}, 出力フォルダ: '{output_folder}'")
    logger.info(f"出力フォーマット: {settings['output_format']}")
    logger.info(f"記号削除: {settings['remove_symbols']}, 漢字→ひらがな変換: {text_processor.convert_kanji}") # 実際に有効になったかを確認

    total_files = len(file_paths)
    processed_count = 0
    error_count = 0

    for i, filepath in enumerate(file_paths):
        logger.info(f"--- 処理中 ({i+1}/{total_files}): {filepath} ---")
        try:
            normalized_filepath = os.path.normpath(filepath)
            if not os.path.isfile(normalized_filepath):
                 logger.warning(f"ファイルが見つからないか、ファイルではありません。スキップします: '{normalized_filepath}'")
                 error_count += 1
                 continue

            process_file(
                normalized_filepath,
                output_folder,
                text_processor,
                settings['output_format']
            )
            processed_count += 1
        except FileProcessingError as e:
            logger.error(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")
            error_count += 1
        except Exception as e:
             logger.error(f"ファイル '{filepath}' の処理中に予期しない重大なエラーが発生しました: {e}", exc_info=True) # スタックトレースも表示
             error_count += 1
        logger.info(f"--- 処理完了 ({i+1}/{total_files}): {filepath} ---")

    # --- 処理結果のサマリ ---
    logger.info("=" * 30 + " 処理結果 " + "=" * 30)
    logger.info(f"合計ファイル数: {total_files}")
    logger.info(f"正常処理ファイル数: {processed_count}")
    logger.info(f"エラー発生ファイル数: {error_count}")
    logger.info(f"出力先フォルダ: {output_folder}")
    logger.info("=" * 68)

    if error_count > 0:
        logger.warning(f"{error_count}個のファイルでエラーが発生しました。詳細はログを確認してください。")
        sys.exit(1) # エラーがあった場合は終了コード1で終了
    else:
        logger.info("すべてのファイルの処理が正常に完了しました。")
        sys.exit(0) # 正常終了

if __name__ == "__main__":
    # スクリプトとして実行された場合にmain関数を呼び出す
    # (pykakasiやtkinterがない場合でもImportErrorは補足されるが、
    #  依存関係が満たされていない旨の警告が出る)
    main()
