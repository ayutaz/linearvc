# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

LinearVC は、自己教師あり音声特徴量（WavLM）に対する線形回帰のみで音声変換（Voice Conversion）を行う研究プロジェクト。Interspeech 2025 に採択された論文の実装コード。

論文: https://arxiv.org/abs/2506.01510

## セットアップ

```bash
conda env create -f environment.yml
conda activate linearvc
```

## 実行コマンド

### 音声変換（非並列データ）
```bash
python linearvc.py \
    --extension .flac \
    <source_wav_dir> <target_wav_dir> <input_wav> output.wav
```

### 音声変換（並列データ）
```bash
python linearvc.py \
    --parallel \
    <source_wav_dir> <target_wav_dir> <input_wav> output.wav
```

### 評価
```bash
# 話者類似度（EER）
python speaker_similarity.py <eval_csv> <converted_dir> <groundtruth_dir>

# 明瞭度（WER/CER）
python intelligibility.py <converted_dir> <groundtruth_dir>
```

### 実験ノートブック
```bash
jupyter lab demo.ipynb                # デモ
jupyter lab experiments_libri.ipynb   # LibriSpeech実験
jupyter lab experiments_vctk.ipynb    # VCTK実験
```

## アーキテクチャ

### コアパイプライン

音声変換は3ステップで行われる:
1. **特徴量抽出**: WavLM Large（レイヤー6）で自己教師あり特徴量を抽出
2. **射影行列の計算**: ソース話者とターゲット話者の特徴量から線形回帰で射影行列 `W` を学習
3. **変換と音声合成**: `Y = X @ W` で特徴量を変換し、HiFiGAN で波形を生成

### 主要モジュール

- **`linearvc.py`**: `LinearVC` クラス（`nn.Module`）— メインの音声変換モジュール兼CLIスクリプト。WavLM と HiFiGAN をラップし、`get_features()`、`get_projmat()`、`project_and_vocode()` の3メソッドで変換を実行
- **`utils.py`**: `fast_cosine_dist()` でフレーム間のコサイン距離マッチング、PCA 変換関数
- **`reduced_rank_ridge.py`**: 縮約ランクリッジ回帰（コンテンツ分解用）— sklearn の `LinearModel` を継承

### 非並列 vs 並列モード

- **非並列（デフォルト）**: コサイン距離マッチングでソース・ターゲット間のフレーム対応を推定。話者あたり約3分のデータが必要
- **並列（`--parallel`）**: 同一ファイル名でペアリング。少量データで動作。デフォルトで Lasso 正則化（α=0.3）を適用

### 回帰手法

- 最小二乗法（`numpy.linalg.lstsq`）— 非並列データのデフォルト
- Lasso 回帰（`celer.Lasso`）— 並列データのデフォルト、`--lasso` で α 値指定
- 縮約ランクリッジ回帰 — コンテンツ分解実験用（ノートブックから使用）

### 外部モデル依存

事前学習モデルは `torch.hub` 経由で `bshall/knn-vc` リポジトリからロードされる（WavLM Large、HiFiGAN）。初回実行時にダウンロードされる。

### グローバル定数

`linearvc.py` 冒頭の `n_frames_max = 8192`（線形回帰の最大フレーム数）と `k_top = 1`（マッチング時の k 値）がハードコードされている。

## 注意事項

- テストスイートは存在しない。検証は実験ノートブックと `log.md` の実験結果で行われている
- リンターやフォーマッター設定はない
- サンプルレートは 16000 Hz 固定
- GPU（CUDA）推奨、CPU フォールバック可能
