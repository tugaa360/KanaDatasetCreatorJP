# KanaDatasetCreatorJP
このモデルは日本語の音声ツールの読み上げの精度向上を目的とした、
漢字かな交じり文を、カタカナ、ひらがなに変換するものです。
出来上がったデータセットは.tsvファイルとして保存されます。
今のところまだ不完全で、一部の漢字やカタカナが上手くデータになっていないところがあります。
このコードの改善やデータセットを多く作り、日本語音声読み上げの向上につながればと思います。

## 依存ライブラリのインストール

このコードを実行するには、以下のライブラリが必要です。

```bash
pip install janome jaconv

コードの説明

    load_texts.py: メインの処理スクリプトです。
    extract_hiragana(text): テキストからひらがなを抽出します。
    preprocess_text(text): テキストの前処理（改行、空白の正規化、カタカナ変換）を行います。
    read_text_with_bom_removal(filepath, encoding='utf-8'): BOM付きの可能性のあるテキストファイルを読み込みます。
    output_comparison_data(filename, original_text, preprocessed_text, hiragana_text, output_folder): 比較結果を TSV ファイルに出力します。
    process_file(filename, input_folder, output_folder): 個々のテキストファイルを処理します。
    load_text_files(folder_path): 指定されたフォルダ内の .txt ファイルのリストを取得します。
