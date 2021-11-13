# Flow Prediction with Deep Neural Network


## 概要
　数値流体計算の代理モデル（Surrogate model）をDeep Neural Networlを用いて作成する。



## 環境
どのOS・ライブラリを使うのか，その構築方法


```
conda create -n fpdnn python=3.8
conda activate fpdnn
```


```
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c anaconda pillow
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge torchinfo
conda install -c conda-forge onnx
```


```
Python 3.8.11
NumPy 1.20.3
matplotlib 3.4.3
PIL 8.0.0
PyTorch 1.9.0
```

```
OpenFOAM 9 (Foundation)
```

See [here](https://openfoam.org/download/9-ubuntu/) for more details on install a OpenFOAM.


## 使い方
何を入力して、何が得られるのか



## 手順

 1. 学習用データの生成 (data)
 2. ニューラルネットワークの学習 (train)
 3. 学習済みニューラルネットワークの利用 (app)



### 1. 学習用データの生成

　学習用データはOpenFOAMを使って生成する。

|項目|	概要|
 - train dataset : 分類器のパラメータを更新するための学習用データ。（ニューラルネットワークだと重みを更新）
 - validation dataset : 手動で設定するパラメータの良し悪しを確かめるための検証用データ。学習は行わない。（ニューラルネットワークだと各層のニューロン数、隠れ層の数、バッチサイズ、学習係数など。ニューラルネットワークの重みは自動更新されるのでハイパーパラメータには含まれない）
 - test dataset : 学習後に汎化性能を確かめるために、最後に（理想的には一度だけ）テストデータで、学習は行わない。

datasetの分割には`torch.utils.data.random_split`を使う。




#### 1.1 Caseファイル

 OpenFOAM Tutorialsのairfoil2Dをベースにする。
 1. systemディレクトリにinternalProbesを追加。
 2. sysytem/controlDict　に functions{ #includeFunc internalProbes}　を追記。
 3. constant/polyMeshを削除。

 [case_template]をコピーして使う


 - [0] Binary mask for shape boundary
 - [1] Freestream field X + shape boundary
 - [2] Freestream field Y + shape boundary
 - [3] Pressure contour
 - [4] X-directional flow velocity contour
 - [5] Y-directional flow velocity contour

[6.3 Sampling and monitoring data](https://cfd.direct/openfoam/user-guide/v9-graphs-monitoring/#x33-2600006.3.1)


### 2. ニューラルネットワークの学習

入力画像データ（3ch x 128 x128）
 - [0] Freestream field X + boundary
 - [1] Freestream field Y + boundary
 - [2] Binary mask for boundary

出力画像データ（3ch x 128 x128）
 - [3] Pressure contour
 - [4] X-directional flow velocity contour
 - [5] Y-directional flow velocity contour



#### 2.1 Neural network archtecture

ネットワークはU-Netをベースにする。


符号化部では、7つの畳み込みブロックを用いる。
符号化層の活性化関数には傾きが0.2のLeakyReLU、復号化層には通常のReLUを用いる。




損失  
[L1LOSS](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)





#### 2.10 ONNX Export


onnxにExportすると　 [Conv] と [Batchnormalization]が統合される。
([ONNX] Add pass that fuses Conv and BatchNormalization #40547)[https://github.com/pytorch/pytorch/pull/40547]





### 3. 学習済みニューラルネットワークの利用

[ONNX Runtime Web（ORT Web）](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-script-tag)




## 設定
どういう設定項目があるのか


## ライセンス
MITライセンスを使う


## 引用のためのbibtexフォーマットのテキスト



## References

[torchinfo](https://github.com/TylerYep/torchinfo)  

[Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations of Airfoil Flows](https://arxiv.org/abs/1810.08217)  

[Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction)
