# XACLE Challenge 2027 個人開発方針

作成日: 2026-10-29。`kyoshioka/model-dev`（個人fork、upstream: `XACLE-Challenge/XACLE_2027_baselinemodel`）での開発方針メモ。
まだコードには反映していない、方針レベルの記録。

## 0. 前提: チームリポジトリとの関係

- [`tech-XACLE-2027`](https://github.com/kmsysgi/tech-XACLE-2027)（Ryo Moritaさんの個人検討リポジトリ、以後「チームリポジトリ」）に、公式ルールの保存・EDA・設計方針メモ・Streamlit分析アプリがまとまっている。
- このリポジトリ（`XACLE_2027_baselinemodel`の個人fork、`kyoshioka/model-dev`ブランチ）は、チームリポジトリの`main`は変更せず、モデルコードの改造だけを独立して進めるための場所。
- チームリポジトリの`apps/data_viewer`（`XACLE_2027_baselinemodel/chkpt/<実験名>/`に`best_model.pt`+`config.json`があれば自動認識）を使うため、ここで学習した実験の`chkpt/<名前>/`をチームリポジトリ側にコピー/symlinkすれば、コード変更なしで比較・診断ができる。

## 1. baseline動作確認済み（2026-10-29実施）

配布済み学習済み重み・スクラッチ学習(20epoch)の両方で、READMEの公式値を再現できることを確認済み。

| | SRCC | LCC | KTAU | MSE |
|---|---:|---:|---:|---:|
| Validation (README公式) | 0.5844 | 0.5995 | 0.4186 | 3.6368 |
| Validation (スクラッチ学習で再現) | 0.5845 | 0.5994 | 0.4187 | 3.6368 |
| Test (README公式) | 0.5648 | 0.6142 | 0.3997 | 3.1847 |
| Test (スクラッチ学習で再現) | 0.5653 | 0.6147 | 0.4001 | 3.1814 |

環境構築の注意点: 社内プロキシのTLS中継が原因でHuggingFaceからのダウンロード時にSSL証明書エラーが出る。
`REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` を設定すれば解決する。

## 2. `XACLE解説.html`（2026全提出手法の検証資料）から分かった重要事実

- **提出要件**: 予測スコア(csv)・システム記述(yaml)・技術レポート(pdf)、**1人最大4システム**。評価データ公開2026-11-19、提出締切2026-12-03。評価データの生成モデル・自然音の出典は意図的に未公表(unseen generator汎化を測るための設計)。
- **アンサンブル禁止の実質的な境界**: 2026上位勢の多くは複数モデル出力の加重平均(Takano_UTokyo_03の18モデル平均、Chunarkar_NTHUのSVR融合、Huang_WHUの"EnTA-Align"等)で戦っていたが、これは2027ルール7("Ensembling results from multiple models is prohibited")では使えない。一方、**Vedula_ASML**の手法は「音声4つ・テキスト3つのエンコーダの埋め込みを**1つのネットワーク**に入れる」方式で、本資料の注記はこれを出力平均と明確に区別している。→ **複数のfrozenエンコーダの特徴を1つの学習ヘッドに入れるのは、2027ルール下でも legitimate な「単一モデル」** と解釈できる具体的先行事例。ただし最終的な解釈は主催者判断なので、大規模にやる前に確認するのが安全。
- **現在のbaseline(M2D-CLAP, val SRCC 0.5844)は2026実績と比べて既に強い**: 2026の各種提出のvalidation SRCCは0.57〜0.62に密集。手法の複雑さだけでSRCCが大きく伸びる領域ではなさそう。2027の勝負所は「未知generatorへの汎化」という新設計そのもの。
- 複数チームが「CLAP/M2D-CLAP埋め込みのコサイン類似度は人間評価と弱い相関」と報告(Ryo Moritaさんのeda004/5の結論と整合)。

## 3. baselineコードのインターフェース(確認済み、`models/xacle_baseline_model.py` 等)

```text
encoder.encode_audio(wavs) -> (B, 768)    # 固定長ベクトル
encoder.encode_text(captions) -> (B, 768) # 固定長ベクトル
  -> L2normalize -> concat(1536) -> Projector(MLP) -> ScorePredictor(Linear->1)
Loss = ClippedMSE(クラス不均衡重み付き) + ContrastiveLoss(バッチ内全ペアのrank margin loss)
```

- エンコーダの差し替えが構造的に疎結合: `encode_audio`/`encode_text`が固定長ベクトルを返せば、Projector・ScorePredictor・Loss・学習ループは変えずに済む。
- `ContrastiveLoss`は既に「バッチ内の全ペアでpred差分とgt差分を合わせる」ランキング損失を持っている。CSB案の核心(同一text×異generatorのペアワイズpreference learning)はゼロから作るものではなく、既存損失の拡張として実装できる。

## 4. VRAM 48GB (RTX 6000 Ada) での見積もり

- Qwen3-Omni-30B-A3B-Instructのtalker(音声生成)・code2wavは**使用禁止事項そのもの**(generation prohibited)なので最初からロードしない。thinker(理解部)だけで十分。
- Ryo Moritaさんのeda002実測: thinker(int4)+talker(bf16)+code2wav(bf16)=27.5GBが24GB GPUに乗らなかった。talker/code2wavを外しthinkerのみ・int8量子化なら、48GBの中で学習ヘッドの活性化メモリも含めて十分な余裕を持ってfrozen特徴抽出ができる。
- thinkerをフルprecision(bf16)かつbackprop/fine-tuningまでやるのは48GBでも厳しい(勾配・オプティマイザ状態を持たせると20〜30B級は溢れる)。
- **結論**: 「30Bを使う」= 「30Bをfine-tuneする」ではない。現在のbaselineと同じ「frozen encoder + 軽量学習ヘッド」のパラダイムを維持したまま、frozenエンコーダの選択肢を大きいモデルまで広げる、というのが48GBの素直な活かし方。

## 5. 方針: モデルだけ置き換えるか、構造を変えるか

**両方やるが、順序が重要。一度に複数の変数を変えない。**

### Stage 0 (他Stageと並行): LOSO評価の自作

公式val/testは既知generatorの範囲内評価であり、2027が測りたい「未知generatorへの汎化」そのものではない(data_viewerの注記にも明記あり)。どの案を選ぶにせよ、trainを割ってgenerator除外評価(Leave-One-Generator-Out)をしないと、本当に汎化しているかは分からない。

### Stage 1: モデル(エンコーダ)だけ置き換える、構造は変えない

`encode_audio`/`encode_text`相当のインターフェースだけ満たす新エンコーダを作り、Projector/ScorePredictor/Lossはbaselineのまま、M2D-CLAPとSRCC(+LOSO)を比較する。エンコーダ単体の優劣だけを見る。

- **特徴抽出方式**: 音声+キャプションを**同一promptに入れ、最終層の隠れ状態を特徴量にする**。eda004/5で「入口の埋め込み(pooling)平均同士の単純比較はほぼ無相関」と分かっているため、「意味的に揃うのはLLM内部(アテンション後の隠れ状態)」というRyo Moritaさんの次の候補に従う。
- **最初に試すbackbone**: 小さいモデルでパイプラインを先に固めてから大きいモデルへ。
  1. **Qwen2.5-Omni-3B**(承認不要リスト記載、約11GBで小さい、joint prompt方式が自然に組める。Ryo Moritaさんのeda004/5レポートの「次の候補」①と一致)でパイプライン(新エンコーダクラス→同一prompt化→隠れ状態抽出→Projector/ScorePredictor/Lossそのまま→LOSO込みの評価)を一通り通す。
  2. 良ければ同じコードパスのまま**Qwen3-Omni-30B-A3B-Instruct**(thinkerのみ、talker/code2wavはロードしない、int8量子化)にスケールする。

### Stage 2: 複数frozenエンコーダの特徴融合(Vedula_ASML方式、構造を少し変える)

Stage 1で複数のbackbone候補(M2D-CLAP、Qwen系等)を単体評価した後、有望なものを1つの学習ヘッドへの入力として連結する。2027ルール上「単一モデル内のfusion」として前例があり安全(出力レベルのアンサンブルとは明確に区別する)。

### Stage 3: 構造そのものを変える(チームリポジトリのidea doc 2案)

backboneが固まった後に重ねる。

1. [評価者ばらつき案](https://github.com/kmsysgi/tech-XACLE-2027/blob/main/docs/idea/XACLE2027_listener_variability_strategy.md): 11クラス分布ヘッド+評価者内ランキング補助損失。backbone非依存で安価。
2. [CSB案](https://github.com/kmsysgi/tech-XACLE-2027/blob/main/docs/idea/XACLE2027_CSB_architecture_implementation.md): 同一text×異generatorのpreference loss、generator-adversarial、latent swap。一番複雑。核(ペアワイズpreference)は既存`ContrastiveLoss`の拡張として実装できる。

## 6. 残っているTODO/確認事項

- [ ] **Ryo Moritaさんにeda006-010の結論を確認する**(未マージブランチ`eda/qwen-audio-zeroshot`。隠れ状態cos-sim／LLM-judge／PMI／describe-then-score。「同一prompt→隠れ状態」方式の確信度に直結。出力が`.gitignore`対象で本人に聞く以外に結論を知る手段がない)。
- [ ] 複数frozenエンコーダの特徴融合(Stage 2)をルール上問題ないか、本格化する前に主催者へ確認するか検討する。
- [ ] Stage 0のLOSO評価ハーネスを実装する。
- [ ] Stage 1のエンコーダ差し替え基盤(Qwen2.5-Omni-3Bから)を実装する。

## 参照

- [公式ルール(チームリポジトリのコピー)](https://github.com/kmsysgi/tech-XACLE-2027/blob/main/docs/official_rules.md)
- [`XACLE解説.html`](../../XACLE解説.html): 2026全提出手法の検証資料
- [評価者ばらつき案](https://github.com/kmsysgi/tech-XACLE-2027/blob/main/docs/idea/XACLE2027_listener_variability_strategy.md)
- [CSB案](https://github.com/kmsysgi/tech-XACLE-2027/blob/main/docs/idea/XACLE2027_CSB_architecture_implementation.md)
- 未マージブランチ: `origin/eda/qwen-audio-zeroshot`(チームリポジトリ、eda004-010)
