import os
import re
import argparse
import configparser
import unicodedata
from typing import List, Optional, Dict, Any, Tuple, Literal
import logging
import json
import sys
import csv
import yaml # PyYAMLが必要: pip install PyYAML
import jaconv # jaconvが必要: pip install jaconv

# --- 依存ライブラリのインポートとオプション化 ---
try:
    from janome.tokenizer import Tokenizer, Token
    from janome.analyzer import Analyzer
    from janome.charfilter import UnicodeNormalizeCharFilter
    from janome.tokenfilter import POSKeepFilter, CompoundNounFilter
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    logging.info("Janomeが見つからないため、読み推定エンジンとしてJanomeは使用できません。")

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False
    logging.info("pyopenjtalkが見つからないため、読み推定エンジンとしてpyopenjtalkは使用できません。")
except Exception as e:
    PYOPENJTALK_AVAILABLE = False
    logging.warning(f"pyopenjtalkのインポート中にエラーが発生しました: {e}。pyopenjtalkは使用できません。")

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
class TextProcessingError(Exception): pass
class FileProcessingError(TextProcessingError): pass
class InitializationError(TextProcessingError): pass
class NormalizationError(TextProcessingError): pass

# --- テキスト正規化クラス ---
class TextNormalizer:
    def __init__(self, rule_filepath: Optional[str] = None):
        self.rules = self._load_rules(rule_filepath)
        logger.info(f"正規化ルールをロードしました (ファイル: {rule_filepath or 'デフォルト'})")
        # 数字変換用の準備など（必要なら）
        self.number_converter = self._create_number_converter() # 例

    def _load_rules(self, filepath: Optional[str]) -> Dict[str, Any]:
        default_rules = {
            'unicode_normalize': 'NFKC',
            'remove_whitespace': True,
            'replace_symbols': { # 読み上げる記号
                '&': ' アンド ', '%': ' パーセント ', '+': ' プラス ', '=':' イコール ',
                '¥': 'エン', '$':'ドル', '€':'ユーロ', '£':'ポンド', # 通貨記号は数字と組み合わせる必要あり
                 '℃':'ド', '℉':'ド', # 温度も数字と組み合わせる
            },
            'remove_symbols': r'[「」『』【】［］（）<>‘’“”・※→←↑↓*＃]', # 発話しない記号
            'punctuation': { # 句読点の扱い
                'remove': True, # Trueなら削除、Falseなら置換
                'replacement': '<pau>', # remove=Falseの場合の置換文字
                'target': '、。？！' # 対象とする句読点
            },
            'number_conversion': { # 数字変換（詳細化が必要）
                 'enabled': True,
                 'target': r'\d+', # マッチさせる正規表現
            },
            'alphabet_conversion': { # アルファベット（詳細化が必要）
                 'enabled': True,
                 'rule': 'spellout', # 'spellout' (ABC->エービーシー) or 'word' (辞書ベース)
                 'dictionary': { 'AI':'エーアイ', 'USB':'ユーエスビー' }
            },
            'custom_replacements': [ # 正規表現による置換 (順番が重要)
                 # {'pattern': r'パターン1', 'replacement': '置換後1'},
                 # {'pattern': r'パターン2', 'replacement': '置換後2'},
            ]
        }
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_rules = yaml.safe_load(f)
                # デフォルトルールをロードしたルールで上書き（マージ）
                # より高度なマージが必要な場合もある
                default_rules.update(loaded_rules)
                return default_rules
            except Exception as e:
                logger.warning(f"正規化ルールファイル '{filepath}' の読み込みに失敗しました: {e}。デフォルトルールを使用します。")
                return default_rules
        else:
            if filepath:
                 logger.warning(f"正規化ルールファイル '{filepath}' が見つかりません。デフォルトルールを使用します。")
            else:
                 logger.info("正規化ルールファイルが指定されていないため、デフォルトルールを使用します。")
            return default_rules

    def _create_number_converter(self):
        # ここで数字を日本語読みに変換するクラスや関数を準備
        # 例: https://github.com/kunisy/python-num2words-ja (要インストール・調整)
        # または自作のルールベース変換器
        # 簡易的な例 (実装が必要)
        class SimpleNumConverter:
            def convert(self, num_str: str) -> str:
                # --- ここに数字を日本語読みにするロジックを実装 ---
                # 例: 123 -> ひゃくにじゅうさん
                # 桁区切り、小数、漢数字、単位なども考慮
                try:
                    # 簡易的に数字をそのまま返す（要実装）
                    num = int(num_str)
                    if num < 10:
                         return ["ゼロ", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ"][num]
                    return num_str # 読み変換ロジックがない場合はそのまま返す
                except ValueError:
                    return num_str # 数字以外はそのまま
        return SimpleNumConverter()


    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
             logger.warning(f"正規化対象が文字列ではありません: {type(text)}")
             return ""
        try:
            # 1. Unicode正規化
            norm_type = self.rules.get('unicode_normalize', 'NFKC')
            if norm_type:
                text = unicodedata.normalize(norm_type, text)

            # 2. カスタム置換 (前処理)
            for rule in self.rules.get('custom_replacements_pre', []):
                text = re.sub(rule['pattern'], rule['replacement'], text)

            # 3. 空白文字の正規化
            if self.rules.get('remove_whitespace', True):
                text = re.sub(r'\s+', ' ', text).strip()

            # 4. 数字変換
            num_conv_conf = self.rules.get('number_conversion', {})
            if num_conv_conf.get('enabled', False):
                 pattern = num_conv_conf.get('target', r'\d+')
                 def num_repl(match):
                     return self.number_converter.convert(match.group(0))
                 text = re.sub(pattern, num_repl, text)

            # 5. アルファベット変換
            alpha_conf = self.rules.get('alphabet_conversion', {})
            if alpha_conf.get('enabled', False):
                # 辞書ベースの置換を優先
                for word, reading in alpha_conf.get('dictionary', {}).items():
                    # 大文字小文字を区別しない場合など、考慮が必要
                    text = text.replace(word, f' {reading} ') # 前後に空白を入れて区切る
                # 残ったアルファベットの処理 (例: スペルアウト)
                if alpha_conf.get('rule') == 'spellout':
                    # --- ここにスペルアウトルールを実装 ---
                    # 例: A -> エー, B -> ビー
                    pass

            # 6. 記号の読み置換
            symbol_map = self.rules.get('replace_symbols', {})
            for symbol, reading in symbol_map.items():
                text = text.replace(symbol, reading) # 前後の空白は読み側で調整

            # 7. 削除する記号
            remove_pattern = self.rules.get('remove_symbols')
            if remove_pattern:
                text = re.sub(remove_pattern, '', text)

            # 8. 句読点の処理
            punct_conf = self.rules.get('punctuation', {})
            target_punct = punct_conf.get('target', '、。？！')
            if punct_conf.get('remove', True):
                text = re.sub(f'[{re.escape(target_punct)}]', '', text)
            else:
                replacement = punct_conf.get('replacement', '<pau>')
                text = re.sub(f'[{re.escape(target_punct)}]', replacement, text)

            # 9. カスタム置換 (後処理)
            for rule in self.rules.get('custom_replacements_post', []):
                 text = re.sub(rule['pattern'], rule['replacement'], text)

            # 10. 再度空白を整理
            if self.rules.get('remove_whitespace', True):
                text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            logger.error(f"テキスト '{text[:50]}...' の正規化中にエラー: {e}", exc_info=True)
            raise NormalizationError(f"Normalization failed: {e}")


# --- 読み・アクセント推定クラス ---
class PronunciationEstimator:
    def __init__(
        self,
        engine: Literal['janome', 'pyopenjtalk'] = 'pyopenjtalk',
        janome_udic_path: Optional[str] = None,
        jtalk_dic_path: Optional[str] = None,
        jtalk_user_dic_path: Optional[str] = None # pyopenjtalk用ユーザー辞書
    ):
        self.engine = engine
        self.jtalk_user_dic_path = jtalk_user_dic_path
        self.unknown_words = set()

        if self.engine == 'janome':
            if not JANOME_AVAILABLE:
                raise InitializationError("Janomeエンジンが指定されましたが、ライブラリが見つかりません。")
            try:
                logger.info(f"Janomeエンジンを初期化します (ユーザー辞書: {janome_udic_path})")
                # 基本的なJanome Analyzer (必要に応じてカスタマイズ)
                char_filters = [UnicodeNormalizeCharFilter()] # NFKC正規化はNormalizer側でも行う
                token_filters = [] # 必要ならCompoundNounFilterなど追加
                self.janome_analyzer = Analyzer(
                    char_filters=char_filters,
                    tokenizer=Tokenizer(udic=janome_udic_path, udic_enc='utf8', wakati=False),
                    token_filters=token_filters
                )
                logger.info("Janomeエンジンの初期化完了。")
            except Exception as e:
                raise InitializationError(f"Janome Analyzerの初期化に失敗しました: {e}")

        elif self.engine == 'pyopenjtalk':
            if not PYOPENJTALK_AVAILABLE:
                raise InitializationError("pyopenjtalkエンジンが指定されましたが、利用できません。")
            try:
                logger.info("pyopenjtalkエンジンを初期化します...")
                if jtalk_dic_path:
                    logger.info(f"pyopenjtalk辞書パスを '{jtalk_dic_path}' に設定します。")
                    os.environ['PYOPENJTALK_DICT_PATH'] = jtalk_dic_path
                # ユーザー辞書オプションを生成
                self.jtalk_opts = []
                if self.jtalk_user_dic_path and os.path.exists(self.jtalk_user_dic_path):
                     # pyopenjtalk.run_frontend は直接ユーザー辞書オプションを取れない？
                     # G2Pのみなら可能かもしれないが、frontendに渡す方法を要調査。
                     # 代替案: MeCabインスタンスを直接操作するか、システム辞書に統合する。
                     # ここでは一旦、警告を出すに留める。
                     logger.warning(f"pyopenjtalkユーザー辞書 '{self.jtalk_user_dic_path}' の直接指定は現在サポートされていません。システム辞書への追加を検討してください。")
                     # self.jtalk_opts = ['-x', jtalk_dic_path, '-u', self.jtalk_user_dic_path] # 例 (動作しない可能性)
                else:
                     if self.jtalk_user_dic_path:
                         logger.warning(f"pyopenjtalkユーザー辞書 '{self.jtalk_user_dic_path}' が見つかりません。")

                _ = pyopenjtalk.g2p('テスト') # 動作確認
                logger.info("pyopenjtalkエンジンの初期化完了。")
            except Exception as e:
                raise InitializationError(f"pyopenjtalkの初期化/動作確認に失敗しました: {e}")
        else:
            raise InitializationError(f"無効なエンジン: {self.engine}")

    def _get_janome_reading(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        tokens_details = []
        katakana_parts = []
        try:
            tokens = self.janome_analyzer.analyze(text)
            for token in tokens:
                surf = token.surface
                pos = token.part_of_speech.split(',')[0]
                reading = token.reading if token.reading != '*' else None
                pron = token.pronunciation if hasattr(token, 'pronunciation') and token.pronunciation != '*' else reading

                if reading:
                     katakana_parts.append(reading)
                elif pron: # 発音があればそれを使う
                     katakana_parts.append(pron)
                else: # 読みがない場合、表層形をカタカナ変換 (簡易)
                     katakana_parts.append(jaconv.hira2kata(surf))
                     # 未知語として記録 (読みがないものは未知語扱いが多い)
                     if pos not in ['記号', '補助記号'] and not re.fullmatch(r'[ぁ-んァ-ンー]+', surf):
                          self.unknown_words.add(surf)

                tokens_details.append({
                    "surface": surf, "pos": token.part_of_speech,
                    "reading": reading, "pronunciation": pron
                })
            return "".join(katakana_parts), tokens_details
        except Exception as e:
            logger.error(f"Janomeでの読み推定中にエラー: {e}", exc_info=True)
            return "[READING ERROR]", []

    def _get_jtalk_pronunciation(self, text: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """pyopenjtalkを使って読み、音素・アクセント情報を取得"""
        try:
            # run_frontend はテキストを受け取り、フルコンテキストラベル文字列のリストを返す
            # 例: ['sil B:^ #-# ...']
            full_context_labels = pyopenjtalk.run_frontend(text, self.jtalk_opts)

            phonemes = []
            accents = [] # アクセント句情報など (詳細なパースが必要)
            katakana_reading_parts = []

            current_mora = ""
            current_accent_val = -1 # アクセント核の位置 (モーラ単位)
            mora_count_in_phrase = 0
            accent_phrase_info = []

            for label in full_context_labels:
                 # ラベルをパースして音素やアクセント情報を抽出
                 # (HTS形式のラベル仕様に基づく: http://hts.sp.nitech.ac.jp/ )
                 # このパースは複雑なので、簡易的な情報抽出に留めるか、専用ライブラリを使う
                 match_phoneme = re.search(r'/A:[^!]*!([^@_]+)@', label) # 基本的な音素
                 match_accent = re.search(r'/F:(\d+)_', label) # アクセント句内のモーラ位置
                 # match_accent_phrase = ... # アクセント句境界などの情報

                 if match_phoneme:
                     ph = match_phoneme.group(1)
                     if ph not in ['sil', 'pau']: # 無音以外を音素リストに追加
                          phonemes.append(ph)
                          # 音素からカタカナ読みを推測 (母音に注目)
                          if ph in 'aiueo':
                              current_mora += ph
                          elif len(ph) > 1 and ph[-1] in 'aiueo': # 子音+母音
                              current_mora += ph[-1]

                          # 簡易的なカタカナ変換 (より正確なマッピングが必要)
                          if ph == 'a': katakana_reading_parts.append('ア')
                          elif ph == 'i': katakana_reading_parts.append('イ')
                          # ... 他の音素のマッピング ...
                          elif ph == 'N': katakana_reading_parts.append('ン')
                          elif ph == 'cl': katakana_reading_parts.append('ッ') # 促音
                          elif len(ph) == 2: # 子音+母音
                              # ここに K+a -> カ のようなマッピングルールが必要
                              pass # 要実装

                 # アクセント情報の簡易抽出 (要改善)
                 if match_accent:
                     accent_in_phrase = int(match_accent.group(1))
                     # アクセント句の区切りや核の位置を特定するロジックが必要

            # pyopenjtalk.g2pを使う方が読み取得は簡単
            katakana_reading = pyopenjtalk.g2p(text, kana=True)

            # 未知語検出（g2pの結果が元のテキストの一部と同じ場合、怪しい）
            if katakana_reading == text and not re.fullmatch(r'[ァ-ンヴー]+', text):
                 self.unknown_words.add(text)

            # run_frontendの結果から詳細なトークン情報は直接取れない
            # MeCabを直接使うか、g2pの結果から推測する必要がある
            tokens_details = [{"surface": text, "reading": katakana_reading}] # 簡易

            return katakana_reading, tokens_details, full_context_labels # LABデータも返す

        except Exception as e:
            logger.error(f"pyopenjtalkでの読み推定中にエラー: {e}", exc_info=True)
            return "[READING ERROR]", [], []


    def get_pronunciation(self, text: str, output_format: Literal['hiragana', 'katakana'] = 'katakana') -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """
        正規化済みテキストの読み、トークン詳細、LABデータを取得します。

        Args:
            text (str): 読みを取得する正規化済みテキスト。
            output_format: 'hiragana' or 'katakana'.

        Returns:
            Tuple[読み(str), トークン詳細(List[Dict]), LABデータ(List[str])]
        """
        reading = ""
        tokens_details = []
        lab_data = []

        if not text:
            return "", [], []

        if self.engine == 'janome':
            reading, tokens_details = self._get_janome_reading(text)
            # JanomeではLABデータは生成されない
        elif self.engine == 'pyopenjtalk':
            reading, tokens_details, lab_data = self._get_jtalk_pronunciation(text)

        # 出力フォーマット変換
        if output_format == 'hiragana' and reading and "[ERROR]" not in reading:
            try:
                reading = jaconv.kata2hira(reading)
            except Exception as e:
                logger.warning(f"ひらがな変換中にエラー: {e}. カタカナのまま返します。")

        return reading, tokens_details, lab_data


# --- ファイルIO & 出力関数 ---
# read_text_with_bom_removal は前回と同じなので省略

def output_corpus_data(
    output_filepath_base: str,
    data: List[Dict[str, Any]], # データ型をAnyに変更
    output_formats: List[Literal['tsv', 'jsonl']]
) -> None:
    """コーパスデータを指定フォーマットで出力"""
    output_successful = []

    if not data:
        logger.warning(f"出力データが空です。{output_filepath_base} 関連ファイルは作成されません。")
        return

    # --- TSV Output ---
    if 'tsv' in output_formats:
        output_tsv_filepath = f"{output_filepath_base}.tsv"
        try:
            # JSONL出力用に準備したデータからTSVに必要なフィールドを抽出
            header = ["id", "text", "reading"] # 基本ヘッダー
            # data[0]に他のキーがあればヘッダーに追加することも可能
            # if "accent_info" in data[0]: header.append("accent_info") # 例

            with open(output_tsv_filepath, 'w', encoding='utf-8', errors='replace', newline='') as tsvfile:
                writer = csv.DictWriter(tsvfile, fieldnames=header, delimiter='\t', quoting=csv.QUOTE_MINIMAL, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(data) # extrasaction='ignore' で不要なキーは無視
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
                    # LABデータは長すぎるのでJSONLには含めないか、別途出力
                    item_copy = item.copy()
                    if 'lab_data' in item_copy:
                         del item_copy['lab_data'] # LABデータはJSONLから削除
                    jsonlfile.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            logger.info(f"コーパスデータをJSONL形式で '{output_jsonl_filepath}' に出力しました。")
            output_successful.append('jsonl')
        except Exception as e:
            logger.error(f"JSONLファイル '{output_jsonl_filepath}' への書き込みに失敗しました: {e}")

    if not output_successful and output_formats:
        raise FileProcessingError(f"指定されたフォーマット ({output_formats}) での出力に成功しませんでした。")


def output_lab_data(output_filepath_base: str, all_lab_data: Dict[str, List[str]]) -> None:
    """LABデータをIDごとに別ファイルに出力"""
    lab_dir = f"{output_filepath_base}_lab"
    try:
        os.makedirs(lab_dir, exist_ok=True)
        for file_id, lab_lines in all_lab_data.items():
            if lab_lines: # LABデータが存在する場合のみファイル作成
                lab_filepath = os.path.join(lab_dir, f"{file_id}.lab")
                with open(lab_filepath, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lab_lines))
        logger.info(f"LABデータを '{lab_dir}' に出力しました。")
    except Exception as e:
        logger.error(f"LABデータの出力中にエラーが発生しました: {e}")


# output_comparison_data は前回とほぼ同じなので省略 (ヘッダーに注意)

# --- ファイル処理関数 ---
def process_file(
    filepath: str,
    output_folder: str,
    normalizer: TextNormalizer,
    pron_estimator: PronunciationEstimator,
    output_formats: List[Literal['tsv', 'jsonl']],
    lab_output_enabled: bool,
    reading_output_format: Literal['hiragana', 'katakana'],
    encoding: str,
    output_comparison: bool = False,
) -> Tuple[int, int]:
    """ファイルを処理し、コーパスデータを出力"""
    filename = os.path.basename(filepath)
    output_base = os.path.join(output_folder, os.path.splitext(filename)[0])
    processed_data: List[Dict[str, Any]] = []
    comparison_items: List[Dict[str, str]] = []
    all_lab_data: Dict[str, List[str]] = {} # LABデータ格納用
    success_count = 0
    error_count = 0

    try:
        # read_text_with_bom_removal は行リストを返すように変更済みと想定
        original_lines = read_text_with_bom_removal(filepath, encoding) # この関数は自前で定義 or 前回のを使う
        logger.info(f"ファイル '{filename}' を読み込みました。行数: {len(original_lines)}")

        for i, line in enumerate(original_lines):
            line_id = f"{os.path.splitext(filename)[0]}_{i:04d}"
            original_line = line.strip()

            if not original_line: continue

            normalized_line = "[NORM ERROR]"
            reading = "[READING ERROR]"
            tokens_details = []
            lab_data = []
            line_success = False

            try:
                normalized_line = normalizer.normalize(original_line)

                if not normalized_line:
                    logger.warning(f"ID {line_id}: 正規化後に空になったためスキップ。 Original: '{original_line[:50]}...'")
                    continue

                reading, tokens_details, lab_data = pron_estimator.get_pronunciation(
                    normalized_line, reading_output_format
                )

                if "[ERROR]" not in reading:
                    line_success = True

            except NormalizationError as e:
                 logger.error(f"ID {line_id}: 正規化エラー: {e}")
            except Exception as e:
                 logger.error(f"ID {line_id}: 行処理エラー: {e}", exc_info=True)

            # --- 結果を格納 ---
            if line_success:
                success_count += 1
            else:
                error_count += 1

            corpus_item = {
                "id": line_id,
                "text": normalized_line, # 正規化後のテキスト
                "reading": reading,
                # "tokens": tokens_details, # 必要なら追加
                # アクセント情報などもここに追加 (要実装)
            }
            if lab_output_enabled and lab_data:
                 corpus_item["lab_data"] = lab_data # 一時的に保持

            processed_data.append(corpus_item)

            if output_comparison:
                comparison_items.append({
                    "id": line_id,
                    "original": original_line,
                    "normalized": normalized_line,
                    "reading": reading
                })

            # LABデータは別管理
            if lab_output_enabled and lab_data:
                all_lab_data[line_id] = lab_data


        # --- ファイル単位で出力 ---
        if processed_data:
            output_corpus_data(output_base, processed_data, output_formats)
        else:
            logger.info(f"ファイル '{filename}' から処理可能なデータなし。")

        if lab_output_enabled and all_lab_data:
            output_lab_data(output_base, all_lab_data)

        if output_comparison and comparison_items:
             # output_comparison_data を呼び出す (前回のを流用 or 修正)
             pass # output_comparison_data(output_base, comparison_items)


        logger.info(f"ファイル '{filename}' 処理完了。成功: {success_count}, エラー: {error_count}")
        return success_count, error_count

    except FileProcessingError as e:
        raise e
    except Exception as e:
        raise FileProcessingError(f"ファイル '{filepath}' 処理中に予期しないエラー: {e}")


# --- メイン関数 ---
def main():
    parser = argparse.ArgumentParser(description='音声合成コーパス向け高度テキスト前処理')

    # 入出力
    parser.add_argument('--input_files', type=str, nargs='+', help='入力ファイルパス (複数可/GUI)')
    parser.add_argument('--output_folder', type=str, help='出力フォルダパス (GUI)')
    parser.add_argument('--output_format', type=str, nargs='+', default=['jsonl'], choices=['tsv', 'jsonl'], help='コーパス出力形式')
    parser.add_argument('--output_lab', action='store_true', help='LAB形式ファイルも出力する (pyopenjtalk必須)')
    parser.add_argument('--output_comparison', action='store_true', help='比較データ(TSV)も出力')
    parser.add_argument('--file_extension', type=str, default='.txt', help='対象拡張子 (GUI用)')
    parser.add_argument('--encoding', type=str, default='utf-8', help='入力エンコーディング')

    # 正規化
    parser.add_argument('--norm_rules', type=str, default='normalization_rules.yaml', help='正規化ルールYAMLファイルのパス')

    # 読み・アクセント推定
    engine_choices = [eng for eng in ['janome', 'pyopenjtalk'] if globals()[f"{eng.upper()}_AVAILABLE"]]
    if not engine_choices: # 利用可能なエンジンがない場合は終了
        parser.error("利用可能な読み推定エンジン(Janome or pyopenjtalk)が見つかりません。")
    parser.add_argument('--engine', type=str, default='pyopenjtalk' if PYOPENJTALK_AVAILABLE else 'janome', choices=engine_choices, help='読み推定エンジン')
    parser.add_argument('--reading_format', type=str, default='katakana', choices=['hiragana', 'katakana'], help='出力する読みの形式')
    parser.add_argument('--janome_udic', type=str, help='Janome用ユーザー辞書パス')
    parser.add_argument('--jtalk_dic', type=str, help='pyopenjtalk用システム辞書パス (環境変数優先)')
    parser.add_argument('--jtalk_user_dic', type=str, help='pyopenjtalk用ユーザー辞書パス (現在、直接指定は実験的)')

    # その他
    parser.add_argument('--config', type=str, help='設定ファイル(.ini)パス (現在未使用)')


    args = parser.parse_args()

    # --- 引数チェック ---
    if args.output_lab and args.engine != 'pyopenjtalk':
        parser.error("--output_lab オプションは --engine pyopenjtalk と共に使用する必要があります。")
    if args.jtalk_user_dic and args.engine != 'pyopenjtalk':
        logger.warning("--jtalk_user_dic は --engine pyopenjtalk でのみ意味があります。")

    # --- 入力/出力パス決定 (GUI対応) ---
    # (前回と同様の select_files_gui, select_output_folder_gui を使う)
    file_paths = args.input_files or (select_files_gui(args.file_extension) if ENABLE_GUI else [])
    if not file_paths:
        logger.error("入力ファイルが指定されていません。")
        sys.exit(1)

    output_folder = args.output_folder or (select_output_folder_gui() if ENABLE_GUI else None)
    if not output_folder:
        logger.error("出力フォルダが指定されていません。")
        sys.exit(1)

    if not os.path.exists(output_folder):
        try: os.makedirs(output_folder); logger.info(f"出力フォルダ作成: {output_folder}")
        except Exception as e: logger.error(f"出力フォルダ作成失敗: {e}"); sys.exit(1)

    output_folder = os.path.normpath(output_folder)
    norm_rules_path = os.path.normpath(args.norm_rules)
    janome_udic_norm = os.path.normpath(args.janome_udic) if args.janome_udic else None
    jtalk_dic_norm = os.path.normpath(args.jtalk_dic) if args.jtalk_dic else None
    jtalk_user_dic_norm = os.path.normpath(args.jtalk_user_dic) if args.jtalk_user_dic else None


    # --- 初期化 ---
    try:
        normalizer = TextNormalizer(norm_rules_path)
        pron_estimator = PronunciationEstimator(
            engine=args.engine,
            janome_udic_path=janome_udic_norm,
            jtalk_dic_path=jtalk_dic_norm,
            jtalk_user_dic_path=jtalk_user_dic_norm
        )
    except InitializationError as e:
        logger.error(f"初期化エラー: {e}"); sys.exit(1)
    except Exception as e:
        logger.error(f"予期せぬ初期化エラー: {e}", exc_info=True); sys.exit(1)

    # --- 処理実行 ---
    logger.info(f"処理開始: {len(file_paths)} ファイル -> {output_folder}")
    # (ログ情報の追加)

    total_success_lines = 0
    total_error_lines = 0
    file_error_count = 0

    for i, filepath in enumerate(file_paths):
        logger.info(f"--- 処理中 ({i+1}/{len(file_paths)}): {filepath} ---")
        try:
            normalized_filepath = os.path.normpath(filepath)
            if not os.path.isfile(normalized_filepath):
                 logger.warning(f"スキップ (ファイルでない): {normalized_filepath}"); file_error_count += 1; continue

            s_count, e_count = process_file(
                normalized_filepath, output_folder, normalizer, pron_estimator,
                args.output_format, args.output_lab, args.reading_format,
                args.encoding, args.output_comparison
            )
            total_success_lines += s_count
            total_error_lines += e_count
            if e_count > 0: file_error_count += 1

        except FileProcessingError as e:
            logger.error(f"ファイルエラー: {e}"); file_error_count += 1
        except Exception as e:
             logger.error(f"予期せぬファイル処理エラー: {e}", exc_info=True); file_error_count += 1
        logger.info(f"--- 処理完了 ({i+1}/{len(file_paths)}): {filepath} ---")

    # --- 未知語の出力 ---
    if pron_estimator.unknown_words:
         unknown_word_file = os.path.join(output_folder, "unknown_words.txt")
         try:
             with open(unknown_word_file, 'w', encoding='utf-8') as f:
                 for word in sorted(list(pron_estimator.unknown_words)):
                     f.write(word + "\n")
             logger.warning(f"未知語の可能性のある単語リストを {unknown_word_file} に出力しました。ユーザー辞書の拡充を検討してください。")
         except Exception as e:
             logger.error(f"未知語リストの出力に失敗しました: {e}")


    # --- 終了処理 & サマリ ---
    logger.info("=" * 30 + " 処理結果 " + "=" * 30)
    # (サマリ情報のログ出力)
    logger.info(f"総処理行数（空行除く）: {total_success_lines + total_error_lines}")
    logger.info(f"  - 正常処理行数: {total_success_lines}")
    logger.info(f"  - エラー発生行数: {total_error_lines}")
    logger.info(f"エラー発生ファイル数: {file_error_count}")
    logger.info("=" * 68)

    sys.exit(1 if file_error_count > 0 or total_error_lines > 0 else 0)


# --- ユーティリティ関数 (read_text_with_bom_removalなど) ---
def read_text_with_bom_removal(filepath: str, encoding: str = 'utf-8') -> List[str]:
    # (前回のコードをここに挿入)
    # ...
    try:
        with open(filepath, 'rb') as f: raw_data = f.read()
        if raw_data.startswith(b'\xef\xbb\xbf'): text = raw_data[3:].decode(encoding, errors='replace')
        elif raw_data.startswith((b'\xff\xfe', b'\xfe\xff')): text = raw_data.decode('utf-16', errors='replace')
        else: text = raw_data.decode(encoding, errors='replace')
        return text.splitlines()
    except FileNotFoundError: raise FileProcessingError(f"ファイルが見つかりません: {filepath}")
    except Exception as e: raise FileProcessingError(f"ファイル読み込みエラー ({filepath}): {e}")

if __name__ == "__main__":
    # 必要なグローバル変数が利用可能かチェック
    if not JANOME_AVAILABLE and not PYOPENJTALK_AVAILABLE:
         print("エラー: Janomeもpyopenjtalkも見つかりません。どちらかが必要です。", file=sys.stderr)
         sys.exit(1)
    main()
