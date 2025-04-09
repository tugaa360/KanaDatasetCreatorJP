import os
import re
import unicodedata
from typing import List, Optional, Dict, Any, Tuple, Literal
import logging
import json
import sys
import csv
import yaml # PyYAMLが必要: pip install PyYAML
import jaconv # jaconvが必要: pip install jaconv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# --- 依存ライブラリのチェックとインポート ---
try:
    from janome.tokenizer import Tokenizer, Token
    from janome.analyzer import Analyzer
    from janome.charfilter import UnicodeNormalizeCharFilter
    # from janome.tokenfilter import POSKeepFilter, CompoundNounFilter # 必要なら使う
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False
except Exception as e: # pyopenjtalkはImportError以外も発生しうる
    PYOPENJTALK_AVAILABLE = False
    print(f"警告: pyopenjtalkのインポート中にエラーが発生しました: {e}", file=sys.stderr)


# --- GUI機能が利用可能か ---
# ENABLE_GUI = True # tkinterのimport成功時にTrueになる想定
try:
    # tkinterは標準ライブラリだが、最小環境でない場合もある
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    ENABLE_GUI = True
except ImportError:
    ENABLE_GUI = False

# ロギングの設定 (GUI表示用にレベルはINFO推奨)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 例外クラス ---
class TextProcessingError(Exception): pass
class FileProcessingError(TextProcessingError): pass
class InitializationError(TextProcessingError): pass
class NormalizationError(TextProcessingError): pass

# --- テキスト正規化クラス ---
class TextNormalizer:
    def __init__(self, rule_filepath: Optional[str] = None):
        self.rules = self._load_rules(rule_filepath)
        logger.info(f"正規化ルールをロードしました (ファイル: {rule_filepath or 'デフォルト'})")
        self.number_converter = self._create_number_converter()

    def _load_rules(self, filepath: Optional[str]) -> Dict[str, Any]:
        # デフォルトルール (必要最低限)
        default_rules = {
            'unicode_normalize': 'NFKC',
            'remove_whitespace': True,
            'replace_symbols': {'&': ' アンド ', '%': ' パーセント '},
            'remove_symbols': r'[「」『』【】［］（）<>‘’“”・※→←↑↓*＃〜]',
            'punctuation': {'remove': False, 'replacement': ' <pau> ', 'target': '、。？！'},
            'number_conversion': {'enabled': True, 'target': r'\d+([\.,]\d+)?'},
            'alphabet_conversion': {'enabled': True, 'rule': 'spellout', 'dictionary': {'AI': 'エーアイ', 'USB': 'ユーエスビー'}},
            'custom_replacements_pre': [],
            'custom_replacements_post': [],
        }
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_rules = yaml.safe_load(f)
                # より安全なマージ (ネストした辞書も考慮するなら再帰的なマージが必要)
                for key, value in loaded_rules.items():
                    if isinstance(value, dict) and isinstance(default_rules.get(key), dict):
                        default_rules[key].update(value)
                    else:
                        default_rules[key] = value
                logger.info(f"カスタム正規化ルール '{filepath}' をロードしました。")
                return default_rules
            except Exception as e:
                logger.warning(f"正規化ルールファイル '{filepath}' の読み込みに失敗: {e}。デフォルトルールを使用。")
                return default_rules
        else:
            if filepath: logger.warning(f"正規化ルールファイル '{filepath}' が見つかりません。デフォルトルールを使用。")
            else: logger.info("正規化ルールファイル未指定。デフォルトルールを使用。")
            return default_rules

    def _create_number_converter(self):
        # 数字 -> 日本語読み 変換クラス (要実装)
        class SimpleNumConverter:
            def convert(self, num_str: str) -> str:
                # ここに詳細な変換ロジックを実装する
                # 例: 123 -> ひゃくにじゅうさん, 10.5 -> じゅってんご
                return num_str # プレースホルダー
        return SimpleNumConverter()

    def normalize(self, text: str) -> str:
        if not isinstance(text, str): return ""
        try:
            # 1. Unicode正規化
            norm_type = self.rules.get('unicode_normalize')
            if norm_type: text = unicodedata.normalize(norm_type, text)

            # 2. 前処理カスタム置換
            for rule in self.rules.get('custom_replacements_pre', []):
                text = re.sub(rule['pattern'], rule['replacement'], text)

            # 3. 空白正規化 (早期に行う方が後続処理が楽な場合も)
            if self.rules.get('remove_whitespace', True):
                text = re.sub(r'\s+', ' ', text).strip()

            # 4. 数字変換 (ルールの target に基づいて検索・置換)
            num_conf = self.rules.get('number_conversion', {})
            if num_conf.get('enabled'):
                 pattern = num_conf.get('target', r'\d+([\.,]\d+)?')
                 try:
                      text = re.sub(pattern, lambda m: self.number_converter.convert(m.group(0)), text)
                 except Exception as e_num: logger.warning(f"数字変換エラー: {e_num}")

            # 5. アルファベット変換 (辞書 + ルール)
            alpha_conf = self.rules.get('alphabet_conversion', {})
            if alpha_conf.get('enabled'):
                for word, reading in alpha_conf.get('dictionary', {}).items():
                    # 大文字小文字区別など考慮
                    text = text.replace(word, f' {reading} ')
                if alpha_conf.get('rule') == 'spellout':
                    # ここにアルファベットを1文字ずつ読む処理 (A->エー) を実装
                    pass

            # 6. 記号読み置換
            for symbol, reading in self.rules.get('replace_symbols', {}).items():
                text = text.replace(symbol, reading)

            # 7. 記号削除
            remove_pattern = self.rules.get('remove_symbols')
            if remove_pattern:
                text = re.sub(remove_pattern, '', text)

            # 8. 句読点処理
            punct_conf = self.rules.get('punctuation', {})
            target_punct = punct_conf.get('target', '、。？！')
            punct_pattern = f'[{re.escape(target_punct)}]'
            if punct_conf.get('remove', True):
                text = re.sub(punct_pattern, '', text)
            else:
                replacement = punct_conf.get('replacement', '<pau>')
                text = re.sub(punct_pattern, replacement, text)

            # 9. 後処理カスタム置換
            for rule in self.rules.get('custom_replacements_post', []):
                 text = re.sub(rule['pattern'], rule['replacement'], text)

            # 10. 最終空白処理
            if self.rules.get('remove_whitespace', True):
                text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            raise NormalizationError(f"正規化失敗 ({text[:20]}...): {e}")


# --- 読み・アクセント推定クラス ---
class PronunciationEstimator:
    def __init__(
        self,
        engine: Literal['janome', 'pyopenjtalk'] = 'pyopenjtalk',
        janome_udic_path: Optional[str] = None,
        jtalk_dic_path: Optional[str] = None,
        jtalk_user_dic_path: Optional[str] = None
    ):
        self.engine = engine
        self.jtalk_user_dic_path = jtalk_user_dic_path # ユーザー辞書パスを保持
        self.unknown_words = set()
        self.janome_analyzer = None # Janomeインスタンス用
        self.jtalk_opts = [] # pyopenjtalkオプション用

        if self.engine == 'janome':
            if not JANOME_AVAILABLE: raise InitializationError("Janome利用不可")
            try:
                logger.info(f"Janome初期化 (ユーザー辞書: {janome_udic_path})")
                char_filters = [UnicodeNormalizeCharFilter()]
                # ユーザー辞書はTokenizerに渡す
                tokenizer_kwargs = {"udic": janome_udic_path, "udic_enc": "utf8"} if janome_udic_path else {}
                self.janome_analyzer = Analyzer(
                    char_filters=char_filters,
                    tokenizer=Tokenizer(**tokenizer_kwargs),
                    token_filters=[] # 必要なら追加
                )
                logger.info("Janome初期化完了")
            except Exception as e: raise InitializationError(f"Janome初期化失敗: {e}")

        elif self.engine == 'pyopenjtalk':
            if not PYOPENJTALK_AVAILABLE: raise InitializationError("pyopenjtalk利用不可")
            try:
                logger.info("pyopenjtalk初期化...")
                if jtalk_dic_path:
                    logger.info(f"jtalkシステム辞書パス設定: {jtalk_dic_path}")
                    os.environ['PYOPENJTALK_DICT_PATH'] = jtalk_dic_path
                # pyopenjtalkのユーザー辞書指定はrun_frontendでは直接できないため注意喚起
                if self.jtalk_user_dic_path:
                     logger.warning(f"jtalkユーザー辞書 '{self.jtalk_user_dic_path}' はrun_frontendでは直接使用されません。システム辞書への統合等を検討してください。g2pには影響する可能性があります。")
                     # g2p に渡すオプションなど、より詳細な制御が必要なら実装

                _ = pyopenjtalk.g2p('テスト') # 動作確認
                logger.info("pyopenjtalk初期化完了")
            except Exception as e: raise InitializationError(f"pyopenjtalk初期化失敗: {e}")
        else:
            raise InitializationError(f"無効なエンジン: {self.engine}")

    def _get_janome_reading(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        tokens_details = []
        katakana_parts = []
        try:
            tokens = self.janome_analyzer.analyze(text)
            for token in tokens:
                surf = token.surface
                pos = token.part_of_speech # 詳細な品詞
                reading = token.reading if token.reading != '*' else None
                pron = token.pronunciation if hasattr(token, 'pronunciation') and token.pronunciation != '*' else reading

                reading_used = reading or pron # 読み or 発音
                if reading_used:
                     katakana_parts.append(reading_used)
                else:
                     katakana_parts.append(jaconv.hira2kata(surf)) # 表層形をカタカナ化
                     if not re.fullmatch(r'[ぁ-んァ-ンー\s]+', surf) and '記号' not in pos:
                         self.unknown_words.add(surf)

                tokens_details.append({
                    "surface": surf, "pos": pos, "reading": reading, "pronunciation": pron
                })
            return "".join(katakana_parts), tokens_details
        except Exception as e:
            logger.error(f"Janome読み推定エラー: {e}", exc_info=True)
            return "[READING ERROR]", []

    def _get_jtalk_pronunciation(self, text: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        katakana_reading = "[READING ERROR]"
        tokens_details = []
        full_context_labels = []
        try:
            # 読みはg2pで取得するのが確実
            # ユーザー辞書はg2pには影響するはず (-u オプション?) 要調査
            g2p_opts = ['-u', self.jtalk_user_dic_path] if self.jtalk_user_dic_path and os.path.exists(self.jtalk_user_dic_path) else []
            # katakana_reading = pyopenjtalk.g2p(text, kana=True, opts=g2p_opts) # opts引数は存在しない？
            katakana_reading = pyopenjtalk.g2p(text, kana=True) # 現状 opts なしで実行

            # LABデータはrun_frontendで取得
            # run_frontendにはユーザー辞書オプションがない
            full_context_labels = pyopenjtalk.run_frontend(text, self.jtalk_opts)

            # 簡易的なトークン情報（本来はMeCabの結果をパースすべき）
            tokens_details = [{"surface": text, "reading": katakana_reading}]

            # 未知語判定（読みが表層形と同じでカタカナのみでない場合）
            if katakana_reading == text and not re.fullmatch(r'[ァ-ンヴー\s]+', text):
                 self.unknown_words.add(text)

            return katakana_reading, tokens_details, full_context_labels
        except Exception as e:
            logger.error(f"pyopenjtalk読み推定エラー: {e}", exc_info=True)
            return "[READING ERROR]", [], []

    def get_pronunciation(self, text: str, output_format: Literal['hiragana', 'katakana'] = 'katakana') -> Tuple[str, List[Dict[str, Any]], List[str]]:
        reading = "[READING ERROR]"
        tokens_details = []
        lab_data = []

        if not text: return "", [], []

        if self.engine == 'janome':
            reading, tokens_details = self._get_janome_reading(text)
        elif self.engine == 'pyopenjtalk':
            reading, tokens_details, lab_data = self._get_jtalk_pronunciation(text)

        # 出力形式変換
        if output_format == 'hiragana' and "[ERROR]" not in reading:
            try: reading = jaconv.kata2hira(reading)
            except Exception: logger.warning("ひらがな変換失敗", exc_info=True)

        return reading, tokens_details, lab_data


# --- ファイルIO & 出力関数 ---
def read_text_with_bom_removal(filepath: str, encoding: str = 'utf-8') -> List[str]:
    try:
        with open(filepath, 'rb') as f: raw_data = f.read()
        if raw_data.startswith(b'\xef\xbb\xbf'): text = raw_data[3:].decode(encoding, errors='replace')
        elif raw_data.startswith((b'\xff\xfe', b'\xfe\xff')): text = raw_data.decode('utf-16', errors='replace')
        else: text = raw_data.decode(encoding, errors='replace')
        return text.splitlines()
    except FileNotFoundError: raise FileProcessingError(f"ファイルが見つかりません: {filepath}")
    except Exception as e: raise FileProcessingError(f"ファイル読み込みエラー ({filepath}): {e}")

def output_corpus_data(output_filepath_base: str, data: List[Dict[str, Any]], output_formats: List[Literal['tsv', 'jsonl']]) -> None:
    output_successful = []
    if not data: logger.warning(f"{output_filepath_base}*: 出力データなし"); return

    # TSV Output
    if 'tsv' in output_formats:
        fpath = f"{output_filepath_base}.tsv"
        try:
            header = ["id", "text", "reading"] # 基本
            # 必要なら他のキーも追加
            with open(fpath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header, delimiter='\t', quoting=csv.QUOTE_MINIMAL, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"TSV出力完了: {fpath}")
            output_successful.append('tsv')
        except Exception as e: logger.error(f"TSV出力失敗 ({fpath}): {e}")

    # JSONL Output
    if 'jsonl' in output_formats:
        fpath = f"{output_filepath_base}.jsonl"
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                for item in data:
                    item_copy = {k: v for k, v in item.items() if k != 'lab_data'} # LABは除く
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            logger.info(f"JSONL出力完了: {fpath}")
            output_successful.append('jsonl')
        except Exception as e: logger.error(f"JSONL出力失敗 ({fpath}): {e}")

    if not output_successful and output_formats:
        raise FileProcessingError("指定形式での出力に成功せず")

def output_lab_data(output_filepath_base: str, all_lab_data: Dict[str, List[str]]) -> None:
    lab_dir = f"{output_filepath_base}_lab"
    try:
        os.makedirs(lab_dir, exist_ok=True)
        count = 0
        for file_id, lab_lines in all_lab_data.items():
            if lab_lines:
                fpath = os.path.join(lab_dir, f"{file_id}.lab")
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lab_lines))
                count += 1
        logger.info(f"LABデータ出力完了: {count}ファイル -> {lab_dir}")
    except Exception as e: logger.error(f"LABデータ出力失敗: {e}")

def output_comparison_data(output_filepath_base: str, comparison_data: List[Dict[str, str]]) -> None:
    """オプション: 元テキスト、正規化テキスト、読みの比較データをTSVで出力"""
    fpath = f"{output_filepath_base}_comparison.tsv"
    try:
        if not comparison_data: return
        header = ["id", "original", "normalized", "reading"]
        with open(fpath, 'w', encoding='utf-8', errors='replace', newline='') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(comparison_data)
        logger.info(f"比較データ出力完了: {fpath}")
    except Exception as e:
        logger.error(f"比較データ出力失敗 ({fpath}): {e}")


# --- ファイル処理関数 ---
def process_file(
    filepath: str, output_folder: str, normalizer: TextNormalizer, pron_estimator: PronunciationEstimator,
    output_formats: List[Literal['tsv', 'jsonl']], lab_output_enabled: bool, reading_output_format: Literal['hiragana', 'katakana'],
    encoding: str, output_comparison: bool = False,
) -> Tuple[int, int]:
    """ファイルを処理し、コーパスデータを出力"""
    filename = os.path.basename(filepath)
    output_base = os.path.join(output_folder, os.path.splitext(filename)[0])
    processed_data: List[Dict[str, Any]] = []
    comparison_items: List[Dict[str, str]] = []
    all_lab_data: Dict[str, List[str]] = {}
    success_count = 0
    error_count = 0

    try:
        original_lines = read_text_with_bom_removal(filepath, encoding)
        logger.debug(f"'{filename}'読み込み完了、{len(original_lines)}行")

        for i, line in enumerate(original_lines):
            line_id = f"{os.path.splitext(filename)[0]}_{i:04d}"
            original_line = line.strip()
            if not original_line: continue

            normalized_line = "[NORM ERROR]"
            reading = "[READING ERROR]"
            lab_data = []
            line_success = False

            try:
                normalized_line = normalizer.normalize(original_line)
                if not normalized_line: logger.warning(f"{line_id}: 正規化後空"); continue

                reading, _, lab_data = pron_estimator.get_pronunciation(normalized_line, reading_output_format)
                if "[ERROR]" not in reading: line_success = True

            except NormalizationError as e_norm: logger.error(f"{line_id}: 正規化エラー: {e_norm}")
            except Exception as e_line: logger.error(f"{line_id}: 行処理エラー: {e_line}", exc_info=True)

            # 結果格納
            if line_success: success_count += 1
            else: error_count += 1

            corpus_item = {"id": line_id, "text": normalized_line, "reading": reading}
            processed_data.append(corpus_item)
            if lab_output_enabled and lab_data: all_lab_data[line_id] = lab_data
            if output_comparison:
                comparison_items.append({"id": line_id, "original": original_line, "normalized": normalized_line, "reading": reading})

        # ファイル単位出力
        if processed_data: output_corpus_data(output_base, processed_data, output_formats)
        if lab_output_enabled and all_lab_data: output_lab_data(output_base, all_lab_data)
        if output_comparison and comparison_items: output_comparison_data(output_base, comparison_items)

        logger.debug(f"'{filename}'処理完了 S:{success_count}, E:{error_count}")
        return success_count, error_count

    except FileProcessingError as e_fp: raise e_fp # 再throw
    except Exception as e_f: raise FileProcessingError(f"ファイル処理中エラー ({filepath}): {e_f}")


# --- GUI アプリケーションクラス ---
class TextProcessorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        # --- ウィンドウ設定 ---
        self.title("音声合成コーパス前処理ツール")
        self.geometry("850x700")
        style = ttk.Style(self)
        try: style.theme_use('clam') # or 'alt', 'default', 'classic'
        except tk.TclError: pass # テーマが存在しない場合

        # --- 変数定義 ---
        self.input_files_var = tk.StringVar(value="")
        self.output_folder_var = tk.StringVar(value="")
        self.norm_rules_var = tk.StringVar(value="normalization_rules.yaml") # デフォルトファイル名
        self.janome_udic_var = tk.StringVar(value="")
        self.jtalk_dic_var = tk.StringVar(value="")
        self.jtalk_user_dic_var = tk.StringVar(value="")
        # 利用可能なエンジンのみ選択肢に含める
        engine_choices = [eng for eng in ['janome', 'pyopenjtalk'] if globals()[f"{eng.upper()}_AVAILABLE"]]
        default_engine = 'pyopenjtalk' if PYOPENJTALK_AVAILABLE else ('janome' if JANOME_AVAILABLE else '')
        self.engine_var = tk.StringVar(value=default_engine)
        self.output_format_vars = {"tsv": tk.BooleanVar(value=False), "jsonl": tk.BooleanVar(value=True)}
        self.reading_format_var = tk.StringVar(value="katakana")
        self.output_lab_var = tk.BooleanVar(value=False)
        self.output_comp_var = tk.BooleanVar(value=False)
        self.encoding_var = tk.StringVar(value="utf-8")
        self.processing_active = False # 処理中フラグ

        # --- ウィジェット作成 & ロギング設定 ---
        self._create_widgets()
        self._setup_logging()

        # --- エンジン利用不可の場合の警告 ---
        if not PYOPENJTALK_AVAILABLE:
             logger.warning("pyopenjtalkが見つからないか初期化に失敗しました。pyopenjtalkエンジンは選択できません。")
        if not JANOME_AVAILABLE:
             logger.warning("Janomeが見つかりません。Janomeエンジンは選択できません。")
        if not engine_choices:
             messagebox.showerror("致命的エラー", "利用可能な読み推定エンジン(Janome/pyopenjtalk)がありません。\nライブラリのインストールを確認してください。")
             self.destroy() # アプリケーション終了

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        main_frame.columnconfigure(1, weight=1) # 2列目が伸びるように

        row_idx = 0

        # --- 入出力パス ---
        io_frame = ttk.LabelFrame(main_frame, text=" 入出力 ", padding="10")
        io_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=(0,5))
        io_frame.columnconfigure(1, weight=1)
        ttk.Label(io_frame, text="入力ファイル:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(io_frame, textvariable=self.input_files_var).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(io_frame, text="選択...", width=8, command=self._select_input_files).grid(row=0, column=2, padx=(0,5), pady=2)
        ttk.Label(io_frame, text="出力フォルダ:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(io_frame, textvariable=self.output_folder_var).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(io_frame, text="選択...", width=8, command=self._select_output_folder).grid(row=1, column=2, padx=(0,5), pady=2)
        row_idx += 1

        # --- 設定ファイル ---
        config_frame = ttk.LabelFrame(main_frame, text=" 設定ファイル ", padding="10")
        config_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        config_frame.columnconfigure(1, weight=1)
        ttk.Label(config_frame, text="正規化ルール:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.norm_rules_var).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(config_frame, text="選択...", width=8, command=self._select_norm_rules).grid(row=0, column=2, padx=(0,5), pady=2)
        row_idx += 1

        # --- エンジンと辞書 ---
        engine_frame = ttk.LabelFrame(main_frame, text=" 読み推定エンジンと辞書 ", padding="10")
        engine_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        engine_frame.columnconfigure(1, weight=1)
        engine_choices = [eng for eng in ['janome', 'pyopenjtalk'] if globals()[f"{eng.upper()}_AVAILABLE"]]
        ttk.Label(engine_frame, text="エンジン:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        engine_combo = ttk.Combobox(engine_frame, textvariable=self.engine_var, values=engine_choices, state="readonly", width=15)
        engine_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(engine_frame, text="Janomeユーザー辞書:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(engine_frame, textvariable=self.janome_udic_var).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(engine_frame, text="選択...", width=8, command=lambda: self._select_file(self.janome_udic_var, "Janome辞書", [("CSV", "*.csv")])).grid(row=1, column=2, padx=(0,5), pady=2)

        ttk.Label(engine_frame, text="JTalkシステム辞書:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(engine_frame, textvariable=self.jtalk_dic_var).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(engine_frame, text="選択...", width=8, command=self._select_jtalk_dic).grid(row=2, column=2, padx=(0,5), pady=2)

        ttk.Label(engine_frame, text="JTalkユーザー辞書:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(engine_frame, textvariable=self.jtalk_user_dic_var).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(engine_frame, text="選択...", width=8, command=lambda: self._select_file(self.jtalk_user_dic_var, "JTalkユーザー辞書", [("CSV/DIC", "*.csv *.dic")])).grid(row=3, column=2, padx=(0,5), pady=2)
        row_idx += 1

        # --- 出力オプション ---
        output_frame = ttk.LabelFrame(main_frame, text=" 出力オプション ", padding="10")
        output_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        output_frame.columnconfigure(1, weight=1) # 右側のスペースを確保

        # コーパス形式 (横並び)
        cf_frame = ttk.Frame(output_frame)
        cf_frame.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=2)
        ttk.Label(cf_frame, text="コーパス形式:").pack(side=tk.LEFT, padx=(5,10))
        ttk.Checkbutton(cf_frame, text="TSV", variable=self.output_format_vars["tsv"]).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(cf_frame, text="JSONL", variable=self.output_format_vars["jsonl"]).pack(side=tk.LEFT, padx=5)

        # 読みの形式
        rf_frame = ttk.Frame(output_frame)
        rf_frame.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=2)
        ttk.Label(rf_frame, text="読みの形式:").pack(side=tk.LEFT, padx=(5,10))
        reading_combo = ttk.Combobox(rf_frame, textvariable=self.reading_format_var, values=['katakana', 'hiragana'], state="readonly", width=10)
        reading_combo.pack(side=tk.LEFT, padx=5)

        # その他チェックボックス (横並び)
        oc_frame = ttk.Frame(output_frame)
        oc_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=2)
        ttk.Checkbutton(oc_frame, text="LAB出力 (JTalk)", variable=self.output_lab_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(oc_frame, text="比較データ出力", variable=self.output_comp_var).pack(side=tk.LEFT, padx=5)

        # エンコーディング
        enc_frame = ttk.Frame(output_frame)
        enc_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=2)
        ttk.Label(enc_frame, text="入力エンコーディング:").pack(side=tk.LEFT, padx=(5, 10))
        encoding_entry = ttk.Entry(enc_frame, textvariable=self.encoding_var, width=12)
        encoding_entry.pack(side=tk.LEFT, padx=5)
        row_idx += 1

        # --- 実行ボタン ---
        self.run_button = ttk.Button(main_frame, text="処理実行", command=self._start_processing, style="Accent.TButton") # スタイル適用例
        self.run_button.grid(row=row_idx, column=0, columnspan=3, pady=15)
        row_idx += 1

        # --- ログ表示 ---
        log_frame = ttk.LabelFrame(main_frame, text=" ログ ", padding="10")
        log_frame.grid(row=row_idx, column=0, columnspan=3, sticky="nsew", padx=5, pady=(5,0))
        main_frame.rowconfigure(row_idx, weight=1) # ログエリアが伸縮するように
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, bd=0, relief=tk.FLAT) # ボーダー調整
        self.log_text.grid(row=0, column=0, sticky="nsew")

    # --- ファイル/フォルダ選択メソッド ---
    def _select_input_files(self):
        fpaths = filedialog.askopenfilenames(title="入力ファイル選択 (複数可)", filetypes=[("テキストファイル", "*.txt"), ("全ファイル", "*.*")])
        if fpaths: self.input_files_var.set(";".join(fpaths)) # 区切り文字で結合

    def _select_output_folder(self):
        fpath = filedialog.askdirectory(title="出力フォルダ選択")
        if fpath: self.output_folder_var.set(fpath)

    def _select_file(self, var, title, filetypes):
        fpath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if fpath: var.set(fpath)

    def _select_norm_rules(self):
        self._select_file(self.norm_rules_var, "正規化ルール選択", [("YAML", "*.yaml *.yml"), ("全ファイル", "*.*")])

    def _select_jtalk_dic(self):
         fpath = filedialog.askdirectory(title="JTalkシステム辞書フォルダ選択 (例: .../open_jtalk_dic_utf_8-1.11)")
         if fpath: self.jtalk_dic_var.set(fpath)

    # --- ロギング設定 ---
    def _setup_logging(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
                self.queue = [] # メッセージキュー
                self.processing = False

            def schedule_update(self):
                # 複数メッセージをまとめて更新
                if not self.processing:
                    self.processing = True
                    self.text_widget.after(50, self._update_widget) # 50ms後に更新

            def _update_widget(self):
                self.processing = False
                if not self.text_widget.winfo_exists(): return

                self.text_widget.configure(state='normal')
                while self.queue:
                    record = self.queue.pop(0)
                    msg = self.format(record)
                    self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.yview(tk.END)

            def emit(self, record):
                self.queue.append(record)
                self.schedule_update()

        gui_handler = TextHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(gui_handler)
        logging.getLogger().setLevel(logging.INFO) # INFOレベル以上をGUIに表示

    # --- 処理実行メソッド ---
    def _start_processing(self):
        if self.processing_active: logger.warning("処理中"); return

        # --- 設定値取得 & バリデーション ---
        self.params = {} # パラメータを格納する辞書
        try:
            self.params['input_files'] = [f.strip() for f in self.input_files_var.get().split(';') if f.strip()]
            self.params['output_folder'] = self.output_folder_var.get()
            self.params['norm_rules'] = self.norm_rules_var.get()
            self.params['engine'] = self.engine_var.get()
            self.params['janome_udic'] = self.janome_udic_var.get() or None
            self.params['jtalk_dic'] = self.jtalk_dic_var.get() or None
            self.params['jtalk_user_dic'] = self.jtalk_user_dic_var.get() or None
            self.params['output_format'] = [fmt for fmt, var in self.output_format_vars.items() if var.get()]
            self.params['reading_format'] = self.reading_format_var.get()
            self.params['output_lab'] = self.output_lab_var.get()
            self.params['output_comparison'] = self.output_comp_var.get()
            self.params['encoding'] = self.encoding_var.get().strip() or 'utf-8'

            if not self.params['input_files']: raise ValueError("入力ファイル未選択")
            if not self.params['output_folder']: raise ValueError("出力フォルダ未選択")
            if not self.params['output_format']: raise ValueError("出力コーパス形式未選択")
            if self.params['output_lab'] and self.params['engine'] != 'pyopenjtalk':
                raise ValueError("LAB出力はpyopenjtalkエンジン選択時のみ可能")
            if not os.path.exists(self.params['norm_rules']):
                logger.warning(f"正規化ルールファイル '{self.params['norm_rules']}' が存在しません。デフォルトルールを使用します。")
                # デフォルトを使用する場合、ファイルパスはNoneとして扱う方が良いかも
                # self.params['norm_rules'] = None # またはそのまま存在しないパスを渡す

        except ValueError as ve:
            messagebox.showerror("入力エラー", str(ve)); return
        except Exception as e:
            messagebox.showerror("設定エラー", f"設定値の取得中にエラー: {e}"); return

        # 処理開始
        self.processing_active = True
        self.run_button.config(state=tk.DISABLED, text="処理中...")
        self.log_text.configure(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.configure(state='disabled')
        logger.info("処理を開始します...")

        # 別スレッドで実行
        self.processing_thread = threading.Thread(target=self._run_processing_logic, daemon=True)
        self.processing_thread.start()
        self.after(100, self._check_thread) # スレッド監視開始

    def _check_thread(self):
        if self.processing_thread.is_alive():
            self.after(100, self._check_thread)
        # スレッド終了後の処理は _run_processing_logic の最後で after(0, ...) を使う

    def _run_processing_logic(self):
        """実際の処理 (別スレッドで実行)"""
        exit_code = 1
        final_message = "処理中にエラーが発生しました。"
        message_type = "error"
        p = self.params # 短縮名

        try:
            # --- ログ表示 (主要パラメータ) ---
            logger.info(f"入力ファイル数: {len(p['input_files'])}")
            logger.info(f"出力フォルダ: {p['output_folder']}")
            logger.info(f"正規化ルール: {p['norm_rules'] if os.path.exists(p['norm_rules']) else 'デフォルト'}")
            logger.info(f"エンジン: {p['engine']}")
            logger.info(f"出力形式: {p['output_format']}")
            logger.info(f"LAB出力: {p['output_lab']}")

            # --- 初期化 ---
            normalizer = TextNormalizer(p['norm_rules'])
            pron_estimator = PronunciationEstimator(
                engine=p['engine'], janome_udic_path=p['janome_udic'],
                jtalk_dic_path=p['jtalk_dic'], jtalk_user_dic_path=p['jtalk_user_dic']
            )

            # --- ファイル処理ループ ---
            total_success_lines, total_error_lines, file_error_count, processed_files = 0, 0, 0, 0
            for i, filepath in enumerate(p['input_files']):
                logger.info(f"--- 処理中 ({i+1}/{len(p['input_files'])}): {os.path.basename(filepath)} ---")
                try:
                    norm_filepath = os.path.normpath(filepath)
                    if not os.path.isfile(norm_filepath):
                         logger.warning(f"スキップ (非ファイル): {norm_filepath}"); file_error_count += 1; continue
                    s, e = process_file( # process_fileは (成功数, エラー数) を返す想定
                        norm_filepath, p['output_folder'], normalizer, pron_estimator,
                        p['output_format'], p['output_lab'], p['reading_format'],
                        p['encoding'], p['output_comparison']
                    )
                    total_success_lines += s; total_error_lines += e
                    if e > 0: file_error_count += 1
                    processed_files += 1
                except FileProcessingError as e_fp: logger.error(f"{os.path.basename(filepath)} ファイルエラー: {e_fp}"); file_error_count += 1
                except Exception as e_f: logger.error(f"{os.path.basename(filepath)} 予期せぬエラー: {e_f}", exc_info=True); file_error_count += 1
                # logger.info(f"--- 処理完了 ({i+1}/{len(p['input_files'])}): {os.path.basename(filepath)} ---") # ログが冗長ならコメントアウト

            # --- 未知語出力 ---
            if pron_estimator.unknown_words:
                 unknown_file = os.path.join(p['output_folder'], "unknown_words.txt")
                 try:
                     with open(unknown_file, 'w', encoding='utf-8') as f:
                         f.write("\n".join(sorted(list(pron_estimator.unknown_words))))
                     logger.warning(f"未知語リスト出力: {unknown_file}")
                 except Exception as e_unk: logger.error(f"未知語リスト出力失敗: {e_unk}")

            # --- 最終結果判定 ---
            logger.info("=" * 30 + " 処理結果 " + "=" * 30)
            logger.info(f"処理ファイル数: {processed_files} / {len(p['input_files'])}")
            logger.info(f"エラー発生ファイル数: {file_error_count}")
            logger.info(f"総処理行数: {total_success_lines + total_error_lines}")
            logger.info(f"  正常処理行数: {total_success_lines}")
            logger.info(f"  エラー発生行数: {total_error_lines}")
            logger.info("=" * 68)

            if file_error_count > 0 or total_error_lines > 0:
                final_message = "処理完了（エラーあり）。詳細はログを確認してください。"
                message_type = "warning"
                exit_code = 1
            else:
                final_message = "すべての処理が正常に完了しました。"
                message_type = "info"
                exit_code = 0

        except InitializationError as e_init: final_message = f"初期化エラー: {e_init}"; message_type = "error"
        except Exception as e_main: final_message = f"予期せぬエラー: {e_main}"; message_type = "error"; logger.error("予期せぬエラー", exc_info=True)

        # --- GUI要素の更新 (メインスレッドで実行) ---
        def update_gui():
            self.processing_active = False
            if self.winfo_exists(): # ウィンドウが閉じられていないか確認
                 self.run_button.config(state=tk.NORMAL, text="処理実行")
                 if message_type == "info": messagebox.showinfo("完了", final_message)
                 elif message_type == "warning": messagebox.showwarning("完了（一部エラー）", final_message)
                 else: messagebox.showerror("エラー", final_message)
        self.after(0, update_gui)
        # スレッド終了


# --- アプリケーション起動 ---
if __name__ == "__main__":
    if not ENABLE_GUI:
         print("エラー: GUI表示に必要な tkinter が見つかりません。", file=sys.stderr)
         sys.exit(1)

    # 利用可能なエンジンがない場合のチェックはGUIクラスの __init__ で行う
    app = TextProcessorGUI()
    app.mainloop()
