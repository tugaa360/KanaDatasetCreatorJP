import os
import re
import argparse
import configparser
import unicodedata
from typing import List, Optional, Dict, Tuple, Literal
import logging
import json
import sys
import csv

# --- 依存ライブラリのインポートとオプション化 ---
try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    logging.info("Janomeが見つからないため、読み推定エンジンとしてJanomeは使用できません。")

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
    # pyopenjtalk内部で使う辞書のパス設定が必要な場合がある
    # 例: os.environ['PYOPENJTALK_DICT_PATH'] = '/path/to/open_jtalk_dic_utf_8-1.11'
except ImportError:
    PYOPENJTALK_AVAILABLE = False
    logging.info("pyopenjtalkが見つからないため、読み推定エンジンとしてpyopenjtalkは使用できません。")
except Exception as e:
    PYOPENJTALK_AVAILABLE = False
    logging.warning(f"pyopenjtalkのインポート中にエラーが発生しました: {e}。pyopenjtalkは使用できません。")


# Janome必須ライブラリ
if JANOME_AVAILABLE:
    import jaconv

# --- GUI機能のオプション化 ---
ENABLE_GUI = True
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:
    ENABLE_GUI = False
    logging.warning("tkinterが見つからないため、GUI機能は無効になります。")

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 例外クラス ---
class TextProcessingError(Exception):
    """テキスト処理に関するカスタム例外クラス"""
    pass

class FileProcessingError(TextProcessingError):
    """ファイル処理に関するカスタム例外クラス"""
    pass

class InitializationError(TextProcessingError):
    """初期化に関するカスタム例外クラス"""
    pass

# --- TextProcessorクラス ---
class TextProcessor:
    def __init__(
        self,
        reading_engine: Literal['janome', 'pyopenjtalk'] = 'janome',
        udic_path: Optional[str] = None,
        g2p_dict_path: Optional[str] = None # pyopenjtalk用辞書パス
    ):
        """
        TextProcessorクラスの初期化

        Args:
            reading_engine: 読み推定に使用するエンジン ('janome' or 'pyopenjtalk').
            udic_path: Janome用ユーザー辞書のパス。
            g2p_dict_path: pyopenjtalk用辞書のパス (環境変数で設定されていない場合)。

        Raises:
            InitializationError: 指定されたエンジンの初期化に失敗した場合。
        """
        self.reading_engine = reading_engine

        if self.reading_engine == 'janome':
            if not JANOME_AVAILABLE:
                raise InitializationError("Janomeエンジンが指定されましたが、ライブラリが見つかりません。")
            try:
                logger.info(f"Janomeエンジンを初期化します (ユーザー辞書: {udic_path})")
                # Janomeは udic_enc='utf8' がデフォルトでない場合があるので明示
                self.tokenizer = Tokenizer(udic=udic_path, udic_enc='utf8') if udic_path else Tokenizer()
                logger.info("Janomeエンジンの初期化完了。")
            except Exception as e:
                raise InitializationError(f"Janome Tokenizerの初期化に失敗しました: {e}")
        elif self.reading_engine == 'pyopenjtalk':
            if not PYOPENJTALK_AVAILABLE:
                raise InitializationError("pyopenjtalkエンジンが指定されましたが、ライブラリが見つからないか、初期化に失敗しました。")
            try:
                logger.info("pyopenjtalkエンジンを初期化します...")
                # 辞書パスが引数で指定されたら環境変数より優先して設定 (暫定的な方法)
                if g2p_dict_path:
                     current_dict_path = os.environ.get('PYOPENJTALK_DICT_PATH')
                     if current_dict_path != g2p_dict_path:
                         logger.info(f"pyopenjtalk辞書パスを '{g2p_dict_path}' に設定します。")
                         os.environ['PYOPENJTALK_DICT_PATH'] = g2p_dict_path
                # 動作確認のための簡単な呼び出し
                _ = pyopenjtalk.g2p('テスト')
                logger.info("pyopenjtalkエンジンの初期化完了。")
            except Exception as e:
                raise InitializationError(f"pyopenjtalkの初期化/動作確認に失敗しました: {e}。辞書パスが正しく設定されているか確認してください。")
        else:
            raise InitializationError(f"無効な読み推定エンジンが指定されました: {self.reading_engine}")

    def normalize_text(self, text: str, remove_punctuation: bool = True, punctuation_replacement: Optional[str] = None) -> str:
        """
        音声合成向けにテキストを正規化します。

        Args:
            text (str): 正規化するテキスト。
            remove_punctuation (bool): 文末以外の句読点（、。）を削除するかどうか。
            punctuation_replacement (Optional[str]): 句読点を置換する文字（例: ポーズ記号）。Noneの場合は削除。

        Returns:
            str: 正規化後のテキスト。
        """
        # 1. Unicode正規化 (NFKC)
        text = unicodedata.normalize('NFKC', text)

        # 2. 空白文字の正規化 (改行、タブなどをスペースに、連続スペースを1つに)
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. 記号の処理 (基本的なルール)
        #    コーパスの要件に応じて詳細化が必要
        #    - 句読点（、。）: 削除または置換
        #    - その他の記号: 基本的に削除（必要なら読みを与えるルールを追加）
        if remove_punctuation:
            if punctuation_replacement is not None:
                 text = text.replace('、', punctuation_replacement).replace('。', punctuation_replacement)
            else:
                 text = text.replace('、', '').replace('。', '')
        # 例: その他の一般的な記号を削除 (拡張が必要)
        text = re.sub(r'[「」『』【】［］（）“”‘’・※→←↑↓*＃]', '', text)
        # 例: 感嘆符、疑問符は保持するかもしれないので、ここでは削除しない

        # 4. 数字、アルファベットの処理 (簡易)
        #    pyopenjtalkはある程度読み上げてくれる。Janomeの場合は別途ルールが必要。
        #    ここでは最低限の処理（例：数字とアルファベットの間のスペース確保など）
        #    text = re.sub(r'([0-9])([ぁ-んァ-ヶー一-龯])', r'\1 \2', text)
        #    text = re.sub(r'([a-zA-Z])([ぁ-んァ-ヶー一-龯])', r'\1 \2', text)

        # 再度空白を整理
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_reading(self, text: str, output_format: Literal['hiragana', 'katakana'] = 'katakana') -> str:
        """
        正規化されたテキストの読みを取得します。

        Args:
            text (str): 読みを取得する正規化済みテキスト。
            output_format (Literal['hiragana', 'katakana']): 出力する読みの形式。

        Returns:
            str: 読み（ひらがな or カタカナ）。
        """
        reading = ""
        if not text:
            return ""

        try:
            if self.reading_engine == 'janome':
                if not JANOME_AVAILABLE: return "[Janome unavailable]"
                tokens = self.tokenizer.tokenize(text)
                katakana_parts = []
                for token in tokens:
                    if token.reading != '*':
                        katakana_parts.append(token.reading)
                    else:
                        # readingがない場合、表層形をカタカナに変換しようと試みる (簡易的)
                        katakana_parts.append(jaconv.hira2kata(token.surface))
                reading = "".join(katakana_parts)

            elif self.reading_engine == 'pyopenjtalk':
                if not PYOPENJTALK_AVAILABLE: return "[pyopenjtalk unavailable]"
                # pyopenjtalk.g2pはデフォルトでカタカナ読みを返す
                reading = pyopenjtalk.g2p(text, kana=True)

            # 出力フォーマットに合わせて変換
            if output_format == 'hiragana' and reading:
                if not JANOME_AVAILABLE: # jaconvのためにJanomeが必要
                     logger.warning("読みのひらがな変換にはJanomeが必要です。カタカナのまま出力します。")
                     return reading
                return jaconv.kata2hira(reading)
            else:
                return reading # カタカナ or 空文字

        except Exception as e:
            logger.error(f"テキスト '{text[:50]}...' の読み推定中にエラーが発生しました (エンジン: {self.reading_engine}): {e}")
            return "[READING ERROR]"


# --- ファイルIO関数 ---
def read_text_with_bom_removal(filepath: str, encoding: str = 'utf-8') -> List[str]:
    """
    テキストファイルを読み込み、BOMを取り除き、行リストとして返します。

    Args:
        filepath (str): ファイルパス。
        encoding (str): ファイルのエンコーディング。

    Returns:
        List[str]: 各行のテキストを含むリスト。

    Raises:
        FileProcessingError: ファイルの読み込みに失敗した場合。
    """
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()

        if raw_data.startswith(b'\xef\xbb\xbf'):
            text = raw_data[3:].decode(encoding, errors='replace')
        elif raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
            try:
                text = raw_data.decode('utf-16', errors='replace')
            except UnicodeDecodeError:
                 logger.warning(f"Detected potential UTF-16 BOM in '{filepath}', but decoding failed. Falling back to '{encoding}'.")
                 text = raw_data.decode(encoding, errors='replace')
        else:
            text = raw_data.decode(encoding, errors='replace')

        return text.splitlines() # 行ごとに分割して返す

    except FileNotFoundError:
        raise FileProcessingError(f"ファイル '{filepath}' が見つかりません。")
    except UnicodeDecodeError as e:
        raise FileProcessingError(f"ファイル '{filepath}' のデコードに失敗しました (encoding: {encoding}): {e}")
    except Exception as e:
        raise FileProcessingError(f"ファイル '{filepath}' の読み込み中に予期しないエラーが発生しました: {e}")

# --- 出力関数 ---
def output_corpus_data(
    output_filepath_base: str,
    data: List[Dict[str, str]],
    output_formats: List[Literal['tsv', 'jsonl']]
) -> None:
    """
    処理結果のコーパスデータを指定されたフォーマットで出力します。

    Args:
        output_filepath_base (str): 出力ファイル名のベース（拡張子なし）。
        data (List[Dict[str, str]]): 出力するデータ。各辞書は 'id', 'text', 'reading' キーを持つ想定。
        output_formats (List[Literal['tsv', 'jsonl']]): 出力フォーマットのリスト。

    Raises:
        FileProcessingError: ファイル書き込みに失敗した場合。
    """
    output_successful = []

    # --- TSV Output ---
    if 'tsv' in output_formats:
        output_tsv_filepath = f"{output_filepath_base}.tsv"
        try:
            with open(output_tsv_filepath, 'w', encoding='utf-8', errors='replace', newline='') as tsvfile:
                if not data: # データがない場合もヘッダーは出力
                    tsvfile.write("id\ttext\treading\n")
                else:
                    # ヘッダーを決定 (data[0]のキーを使用)
                    header = list(data[0].keys())
                    writer = csv.DictWriter(tsvfile, fieldnames=header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()
                    writer.writerows(data)
            logger.info(f"コーパスデータをTSV形式で '{output_tsv_filepath}' に出力しました。")
            output_successful.append('tsv')
        except Exception as e:
            logger.error(f"TSVファイル '{output_tsv_filepath}' への書き込みに失敗しました: {e}")

    # --- JSONL Output ---
    if 'jsonl' in output_formats:
        output_jsonl_filepath = f"{output_filepath_base}.jsonl"
        try:
            with open(output_jsonl_filepath, 'w', encoding='utf-8', errors='replace') as jsonlfile:
                for item in data:
                    jsonlfile.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"コーパスデータをJSONL形式で '{output_jsonl_filepath}' に出力しました。")
            output_successful.append('jsonl')
        except Exception as e:
            logger.error(f"JSONLファイル '{output_jsonl_filepath}' への書き込みに失敗しました: {e}")

    if not output_successful and output_formats:
        raise FileProcessingError(f"指定されたフォーマット ({output_formats}) での出力に成功しませんでした。")


def output_comparison_data(
    output_filepath_base: str,
    comparison_data: List[Dict[str, str]]
) -> None:
    """オプション: 元テキスト、正規化テキスト、読みの比較データをTSVで出力"""
    output_tsv_filepath = f"{output_filepath_base}_comparison.tsv"
    try:
        with open(output_tsv_filepath, 'w', encoding='utf-8', errors='replace', newline='') as tsvfile:
            if not comparison_data:
                 tsvfile.write("id\toriginal\tnormalized\treading\n")
            else:
                header = list(comparison_data[0].keys())
                writer = csv.DictWriter(tsvfile, fieldnames=header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                writer.writerows(comparison_data)
        logger.info(f"比較データをTSV形式で '{output_tsv_filepath}' に出力しました。")
    except Exception as e:
        logger.error(f"比較データTSVファイル '{output_tsv_filepath}' への書き込みに失敗しました: {e}")


# --- ファイル処理関数 ---
def process_file(
    filepath: str,
    output_folder: str,
    text_processor: TextProcessor,
    output_formats: List[Literal['tsv', 'jsonl']],
    reading_output_format: Literal['hiragana', 'katakana'],
    encoding: str,
    output_comparison: bool = False,
    remove_punct: bool = True,
    punct_replace: Optional[str] = None
) -> Tuple[int, int]:
    """
    指定されたテキストファイルを読み込み、正規化、読み推定を行い、コーパスデータを出力します。

    Args:
        filepath (str): 処理するファイルのパス。
        output_folder (str): 出力フォルダのパス。
        text_processor (TextProcessor): テキスト処理を行うオブジェクト。
        output_formats (List): 出力フォーマット ('tsv', 'jsonl')。
        reading_output_format (str): 読みの出力形式 ('hiragana', 'katakana')。
        encoding (str): 入力ファイルのエンコーディング。
        output_comparison (bool): 比較データも出力するかどうか。
        remove_punct (bool): 句読点を削除するかどうか (normalize_textへ渡す)。
        punct_replace (Optional[str]): 句読点の置換文字 (normalize_textへ渡す)。

    Returns:
        Tuple[int, int]: (処理成功行数, エラー行数)

    Raises:
        FileProcessingError: ファイル読み込みや出力の致命的なエラー。
    """
    filename = os.path.basename(filepath)
    output_base = os.path.join(output_folder, os.path.splitext(filename)[0])
    processed_data: List[Dict[str, str]] = []
    comparison_items: List[Dict[str, str]] = []
    success_count = 0
    error_count = 0

    try:
        original_lines = read_text_with_bom_removal(filepath, encoding)
        logger.info(f"ファイル '{filename}' を読み込みました。行数: {len(original_lines)}")

        for i, line in enumerate(original_lines):
            line_id = f"{os.path.splitext(filename)[0]}_{i:04d}" # 例: file01_0000
            original_line = line.strip()

            if not original_line: # 空行はスキップ
                continue

            try:
                normalized_line = text_processor.normalize_text(
                    original_line,
                    remove_punctuation=remove_punct,
                    punctuation_replacement=punct_replace
                )

                if not normalized_line: # 正規化後に空になった場合もスキップ
                    logger.warning(f"ID {line_id}: 正規化後に空になったためスキップします。 Original: '{original_line[:50]}...'")
                    continue

                reading = text_processor.get_reading(normalized_line, reading_output_format)

                if "[READING ERROR]" in reading or "[unavailable]" in reading:
                     error_count += 1
                else:
                     success_count += 1

                # コーパスデータに追加
                processed_data.append({
                    "id": line_id,
                    "text": normalized_line,
                    "reading": reading
                })

                # 比較データに追加 (オプション)
                if output_comparison:
                    comparison_items.append({
                        "id": line_id,
                        "original": original_line,
                        "normalized": normalized_line,
                        "reading": reading
                    })

            except Exception as e:
                logger.error(f"ID {line_id}: 行処理中にエラーが発生しました: {e}. Original: '{original_line[:50]}...'")
                # エラー行のデータを追加（読みはエラーを示す）
                processed_data.append({
                    "id": line_id,
                    "text": original_line, # エラー時は元テキストを残すなど方針を決める
                    "reading": "[PROCESSING ERROR]"
                })
                if output_comparison:
                    comparison_items.append({
                        "id": line_id,
                        "original": original_line,
                        "normalized": "[ERROR]",
                        "reading": "[PROCESSING ERROR]"
                    })
                error_count += 1

        # ファイル単位で出力
        if processed_data:
            output_corpus_data(output_base, processed_data, output_formats)
        else:
            logger.info(f"ファイル '{filename}' から処理可能なデータが見つかりませんでした。出力ファイルは作成されません。")

        if output_comparison and comparison_items:
            output_comparison_data(output_base, comparison_items)

        logger.info(f"ファイル '{filename}' の処理完了。成功: {success_count}行, エラー: {error_count}行")
        return success_count, error_count

    except FileProcessingError as e:
        raise e # 再raiseして呼び出し元で処理
    except Exception as e:
        raise FileProcessingError(f"ファイル '{filepath}' の処理中に予期しないエラーが発生しました: {e}")


# --- GUI選択関数 ---
def select_files_gui(file_extension: str = ".txt") -> List[str]:
    if not ENABLE_GUI: return []
    try:
        root = tk.Tk()
        root.withdraw()
        ftypes = [(f"テキストファイル (*{file_extension})", f"*{file_extension}"), ("すべてのファイル", "*.*")]
        fpaths = filedialog.askopenfilenames(title=f"処理するファイルを選択 ({file_extension})", filetypes=ftypes)
        root.destroy()
        return list(fpaths)
    except Exception as e:
        logger.error(f"ファイル選択ダイアログエラー: {e}")
        return []

def select_output_folder_gui() -> Optional[str]:
    if not ENABLE_GUI: return None
    try:
        root = tk.Tk()
        root.withdraw()
        fpath = filedialog.askdirectory(title="出力フォルダを選択")
        root.destroy()
        return fpath if fpath else None
    except Exception as e:
        logger.error(f"フォルダ選択ダイアログエラー: {e}")
        return None

# --- メイン関数 ---
def main():
    parser = argparse.ArgumentParser(description='音声合成コーパス向けテキスト前処理スクリプト。テキストを正規化し、読みを推定します。')
    parser.add_argument('--input_files', type=str, nargs='+', help='処理する入力テキストファイルのパス（複数指定可）。指定しない場合はGUIで選択（GUI有効時）。')
    parser.add_argument('--output_folder', type=str, help='出力フォルダのパス。指定しない場合はGUIで選択（GUI有効時）。')
    parser.add_argument('--config', type=str, help='設定ファイル (.ini) のパス')

    # エンジンと言語リソース
    engine_choices = [eng for eng in ['janome', 'pyopenjtalk'] if globals()[f"{eng.upper()}_AVAILABLE"]]
    if not engine_choices:
         logger.error("利用可能な読み推定エンジン (Janome or pyopenjtalk) が見つかりません。インストールを確認してください。")
         sys.exit(1)
    parser.add_argument('--reading_engine', type=str, default=engine_choices[0], choices=engine_choices, help='読み推定に使用するエンジン。')
    parser.add_argument('--udic_path', type=str, help='Janome用ユーザー辞書 (.csv) のパス')
    parser.add_argument('--g2p_dict_path', type=str, help='pyopenjtalk用辞書のパス (環境変数で設定されていない場合)')

    # 出力設定
    parser.add_argument('--output_format', type=str, nargs='+', default=['jsonl'], choices=['tsv', 'jsonl'], help='コーパスデータの出力フォーマット（複数選択可）')
    parser.add_argument('--reading_output_format', type=str, default='katakana', choices=['hiragana', 'katakana'], help='出力する読みの形式')
    parser.add_argument('--output_comparison', action='store_true', help='元テキスト、正規化テキスト、読みの比較データ(TSV)も出力する')

    # 正規化オプション
    parser.add_argument('--no_remove_punctuation', action='store_true', help='句読点（、。）を削除しない')
    parser.add_argument('--punctuation_replacement', type=str, default=None, help='句読点を削除する代わりにこの文字で置換する (例: <pau>)')

    # その他
    parser.add_argument('--file_extension', type=str, default='.txt', help='処理対象とするファイルの拡張子 (GUI選択時)')
    parser.add_argument('--encoding', type=str, default='utf-8', help='入力ファイルのエンコーディング')

    args = parser.parse_args()

    # --- 設定の読み込み (簡易版) ---
    # configファイルはコマンドライン引数を上書きしない方針とする
    # 必要ならより詳細な優先順位付けを実装
    if args.config and os.path.exists(args.config):
        logger.info(f"設定ファイル '{args.config}' を読み込みますが、コマンドライン引数が優先されます。")
        # config = configparser.ConfigParser()
        # config.read(args.config, encoding='utf-8')
        # ここで config.get などを使ってデフォルト値を設定するロジックを追加可能

    # --- 入力ファイルの決定 ---
    file_paths = args.input_files
    if not file_paths:
        if ENABLE_GUI:
            logger.info("入力ファイルが指定されていないため、GUIで選択します。")
            file_paths = select_files_gui(args.file_extension)
            if not file_paths:
                logger.info("ファイルが選択されませんでした。処理を中止します。")
                sys.exit(0)
        else:
            logger.error("入力ファイルが指定されていません。--input_files 引数で指定してください。")
            sys.exit(1)

    # --- 出力フォルダの決定 ---
    output_folder = args.output_folder
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

    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            logger.info(f"出力フォルダ '{output_folder}' を作成しました。")
        except Exception as e:
            logger.error(f"出力フォルダ '{output_folder}' の作成に失敗しました: {e}")
            sys.exit(1)

    # パスの正規化
    output_folder = os.path.normpath(output_folder)
    udic_path_norm = os.path.normpath(args.udic_path) if args.udic_path else None
    g2p_dict_path_norm = os.path.normpath(args.g2p_dict_path) if args.g2p_dict_path else None

    # --- TextProcessorの初期化 ---
    try:
        text_processor = TextProcessor(
            reading_engine=args.reading_engine,
            udic_path=udic_path_norm,
            g2p_dict_path=g2p_dict_path_norm
        )
    except InitializationError as e:
        logger.error(f"TextProcessorの初期化に失敗しました: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"TextProcessorの初期化中に予期しないエラーが発生しました: {e}")
        sys.exit(1)

    # --- ファイル処理の実行 ---
    logger.info(f"処理を開始します。入力ファイル数: {len(file_paths)}, 出力フォルダ: '{output_folder}'")
    logger.info(f"読み推定エンジン: {args.reading_engine}, 読み出力形式: {args.reading_output_format}")
    logger.info(f"コーパス出力形式: {args.output_format}, 比較データ出力: {args.output_comparison}")
    logger.info(f"句読点削除: {not args.no_remove_punctuation}, 句読点置換文字: {args.punctuation_replacement}")

    total_files = len(file_paths)
    total_success_lines = 0
    total_error_lines = 0
    file_error_count = 0

    for i, filepath in enumerate(file_paths):
        logger.info(f"--- 処理中 ({i+1}/{total_files}): {filepath} ---")
        try:
            normalized_filepath = os.path.normpath(filepath)
            if not os.path.isfile(normalized_filepath):
                 logger.warning(f"ファイルが見つからないか、ファイルではありません。スキップします: '{normalized_filepath}'")
                 file_error_count += 1
                 continue

            s_count, e_count = process_file(
                normalized_filepath,
                output_folder,
                text_processor,
                args.output_format,
                args.reading_output_format,
                args.encoding,
                args.output_comparison,
                not args.no_remove_punctuation,
                args.punctuation_replacement
            )
            total_success_lines += s_count
            total_error_lines += e_count
            if e_count > 0:
                 file_error_count += 1 # 行レベルのエラーがあったファイルもカウント

        except FileProcessingError as e:
            logger.error(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")
            file_error_count += 1
        except Exception as e:
             logger.error(f"ファイル '{filepath}' の処理中に予期しない重大なエラーが発生しました: {e}", exc_info=True)
             file_error_count += 1
        logger.info(f"--- 処理完了 ({i+1}/{total_files}): {filepath} ---")

    # --- 処理結果のサマリ ---
    logger.info("=" * 30 + " 処理結果 " + "=" * 30)
    logger.info(f"合計ファイル数: {total_files}")
    logger.info(f"エラー発生ファイル数: {file_error_count}")
    logger.info(f"総処理行数（空行除く）: {total_success_lines + total_error_lines}")
    logger.info(f"  - 正常処理行数: {total_success_lines}")
    logger.info(f"  - エラー発生行数: {total_error_lines}")
    logger.info(f"出力先フォルダ: {output_folder}")
    logger.info("=" * 68)

    if file_error_count > 0 or total_error_lines > 0:
        logger.warning(f"処理中にエラーが発生しました。詳細はログを確認してください。")
        sys.exit(1)
    else:
        logger.info("すべてのファイルの処理が正常に完了しました。")
        sys.exit(0)

if __name__ == "__main__":
    main()
