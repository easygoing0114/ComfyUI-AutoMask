<div align="center">
<img width="800" height="343" alt="ComfyUI AutoMask Nodes thumbnail" src="images/ComfyUI-AutoMask_banner_image.png">
</div>

# ComfyUI-AutoMask

- [**英語版**](README.md)
- **ガイド（外部サイト）**: [English](https://www.ai-image-journey.com/2026/08/birefnet-depthanythingv3-sam31-lama.html) | [Japanese](https://note.com/ai_image_journey/n/nf45afca51b00)

ComfyUIのカスタムノードです。マスクの境界を自動調整し、ヒストグラムに基づく閾値の候補を自動で算出して、手動での試行錯誤の手間を軽減します。

## ノード

### Mask Refine

<div align="center">
<img width="800" height="666" alt="mask refine node sample" src="images/birefnet_refine_20260808.png">
</div>

**BiRefNet**

<div align="center">
<img width="600" height="305" alt="close up mask and image" src="images/BirRefNet_compare_close_up.png">
</div>

**BiRefNet -> Mask Threshold = 0.01 -> Mask Refine**

<div align="center">
<img width="600" height="305" alt="close up refine mask and image" src="images/BirRefNet_bold_refine_compare_close_up.png">
</div>

マスクを、元の画像をガイドとした閉形式マッティング（[`pymatting`](https://github.com/pymatting/pymatting)）を用いてリファインします。

> このノードは、[spacepxl](https://github.com/spacepxl) 氏の [ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) にある **Image Matting** ノードを基にした簡易実装です。

- **入力**
  - `image` — `IMAGE`、元の画像
  - `mask` — `MASK`、リファイン対象のマスク
  - `preblur` — `INT`（デフォルト `10`）、事前にマスクに適用するガウシアンブラーの半径、値が大きいほど硬い境界が柔らかく整います。`0` でプリブラーを無効化します。
- **出力**: `MASK` — リファインされたマスク

任意の粗いセグメンテーション（SAM 3.1、BiRefNet、手動マスク、閾値処理など）の後に使用すると、特に髪・毛皮・半透明領域などのエッジがきれいに整います。

### Auto Mask Threshold

<div align="center">
<img width="800" height="682" alt="auto mask threshold sample" src="images/auto_mask_threshold_sample_20260808.png">
</div>

マスクのヒストグラムを解析して、意味のある閾値を自動で検出します。

- **入力**
  - `mask` — `MASK`、ソフト（非バイナリ）マスク
  - `threshold` — `INT`（デフォルト `5`、範囲 `1`–`8`）、単一マスク出力用に検出された境界のうち n 番目（昇順）を選択
- **出力**
  - `mask_batch (x8)` — `MASK`、検出された境界ごとのバイナリマスクを8枚のバッチで出力
  - `mask` — `MASK`、`threshold` で選択された単一のバイナリマスク
  - `histogram` — `IMAGE`、検出されたピーク／谷と選択された境界を示したヒストグラム画像（ノードプレビューにも表示）

ヒストグラムのピーク間の谷を最大5つ、さらに最大ピーク内の小さな極小値を最大3つ検出し、最大8つの候補を提示します。異なるカットオフ値を素早く比較するのに便利です。

## インストール

### ComfyUI Manager 経由（推奨）

Nodes Manager で **"easygoing"** と検索してインストールしてください。

<div align="center">
<img width="600" height="278" alt="comfyui extension search easygoing" src="images/comfyui_extension_search_easygoing_with_comment.png">
</div>

### 手動インストール

ComfyUI の Python 仮想環境を有効にした後：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/easygoing0114/ComfyUI-AutoMask.git
cd ComfyUI-AutoMask
pip install -r requirements.txt
```

## 依存ライブラリ

- ComfyUI
- `opencv-python`
- `numpy`
- `pymatting`
- `Pillow`

（`torch` と `folder_paths` は ComfyUI に含まれています。）

## クレジット

- **Mask Refine** は、[spacepxl](https://github.com/spacepxl) 氏の [ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) にある **Image Matting** ノードを簡略化した再実装です。

## ライセンス

[MIT](LICENSE)
