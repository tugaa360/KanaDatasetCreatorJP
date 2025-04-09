# KanaDatasetCreatorJP
このモデルは日本語の音声ツールの読み上げの精度向上を目的とした、
漢字かな交じり文を、カタカナ、ひらがなに変換するものです。
出来上がったデータセットは.tsvファイルとして保存されます。
今のところまだ不完全で、一部の漢字やカタカナが上手くデータになっていないところがあります。
このコードの改善やデータセットを多く作り、日本語音声読み上げの向上につながればと思います。

## 依存ライブラリのインストール

このコードを実行するには、以下のライブラリが必要です。

```bash
pip install -r requirements.txt
もしくは
pip install pyopenjtalk PyYAML jaconv janome

コードの説明
# pyopenjtalkを使用し、JSONLとLAB形式で出力、比較データも出力
python your_script_name.py \
    --input_files input1.txt input2.txt \
    --output_folder ./corpus_output \
    --engine pyopenjtalk \
    --output_format jsonl \
    --output_lab \
    --output_comparison \
    --norm_rules normalization_rules.yaml \
    # --jtalk_dic /path/to/open_jtalk_dic (必要なら)
    # --jtalk_user_dic /path/to/user.dic (現在実験的)

# Janomeを使用し、TSV形式でひらがな読みを出力
python your_script_name.py \
    --input_files data/*.txt \
    --output_folder ./corpus_output_janome \
    --engine janome \
    --output_format tsv \
    --reading_format hiragana \
    # --janome_udic /path/to/janome_user.csv (必要なら)
