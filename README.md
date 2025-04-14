# 栩栩如生：動畫人物動作重演
## 1.簡介

## 2.系統流程
<div align="center">
    <img src="./docs/Lifelike.png" alt="Lifelike的系統流程圖。"/>
</div>
圖 2.1：Lifelike的系統流程圖。

圖 2.1為本系統流程圖，關鍵點檢測器(Keypoint Detector)會提取10組關鍵點集合，再根據每組關鍵點集合計算1個TPS轉換。同時，背景預測器(BG Motion Predictor)會估計1個表示背景運動的仿射轉換。密集動作網路(Dense Motion Network)根據11個轉換的資訊估計多解析度光流(Multi-resolution Optical flows)和多解析度遮擋遮罩(Multi-resolution Occlusion Masks)。修復網路(Inpainting Network)會將來源影像編碼為不同尺度的特徵圖，先使用多解析度光流扭曲相應解析度的特徵圖，再使用多解析度遮擋遮罩修復相應解析度的被扭曲之特徵圖，最後輸出一張動作重演的影像。

## 3.多解析度光流技術

<div align="center">
    <img src="./docs/MROF.png" alt="組合多解析度光流的流程圖。"/>
</div>
圖 3.1：組合多解析度光流的流程圖。

如圖 3.1所示，經由背景預測器和關鍵點檢測器可分別取得1個64×64的仿射轉換(灰色)和10個64×64的TPS轉換(灰色以外的顏色)。本研究使用下采樣(downsampling)和上采樣(upsampling)得到11個32×32轉換、11個128×128轉換和11個256×256轉換。同時，密集動作網路會估計多解析度貢獻圖和多解析度遮擋遮罩，每張貢獻圖可表示對應轉換的權重，透過計算同一解析度中所有轉換和其對應之貢獻圖逐點乘積(Pixel-wise multiplication)的總和，即可得到不同解析度的光流。

## 4.特定型態資料增強
<div align="center">
    <img src="./docs/AO.png" alt="原AnimeCeleb資料集動作影片範例之片段。"/>
</div>
圖 4.1：原AnimeCeleb資料集動作影片範例之片段

<div align="center">
    <img src="./docs/AOD.png" alt="經過特定型態資料增強動作影片範例之片段。"/>
</div>
圖 4.2：經過特定型態資料增強動作影片範例之片段

AnimeCeleb資料集缺少動畫人物肩膀移動的動作資訊(圖 4.1)。由於本系統會從兩張影像之間學習動作轉換，若兩張影像中動畫人物的肩膀在相同的位置，系統會認為動畫人物肩膀不會移動，造成系統不會學習動畫人物肩膀的動作轉換。為能夠讓系統學習動畫人物肩膀移動的動作轉換，本研究藉由移動像素座標的方式模擬動畫人物的肩膀移動(圖 4.2)。

## 5.測試階段
本系統在訓練階會透過同一段影片的兩張影格學習動作轉換，然而在測試階段來源影像和驅動影片的人物身分通常不一致，若直接使用來源關鍵點和驅動關鍵點預測動作轉換，轉換結果易受到外貌的幾何差異影響，導致生成不如預期的動作重演動畫。本研究採用相對模式(relative mode)推算驅動關鍵點在來源影像的相對位置，讓系統可透過來源關鍵點和相對驅動關鍵點預測動作轉換，達到動作重演的理想效果。

<div align="center" style="background-color: white; border: 1px solid white; padding: 10px;">
    <img src="./docs/TestMode.png" alt="本系統測試階段的流程圖。"/>
</div>
圖 5.1：本系統測試階段的流程圖

如圖 5.1所示，本系統會從驅動影片選擇一張與來源影像動作相似的影格，稱之為初始影像(Initial)。關鍵點檢測器會分別從來源影像(Source)、初始影像(Initial)和驅動影像(Driving)提取關鍵點。相對模式會計算驅動關鍵點和初始關鍵點的偏移量，以及計算來源關鍵點和初始關鍵點的凸包面積比，再根據偏移量和凸包面積比推算驅動關鍵點在來源影像的相對位置。在得知相對驅動關鍵點後，系統即可利用來源關鍵點和相對驅動關鍵點計算10個TPS轉換，並預測多解析度光流和多解析度遮擋遮罩，完成後續的動作轉換。

## 6.測試階段
<div align="center">
    <img src="./docs/r1.png" alt="生成動作重演動影像的範例。"/>
</div>
圖 6.1：生成動作重演動影像的範例。
<div align="center">
    <img src="./docs/r2.png" alt="生成重建之來源影像的範例。"/>
</div>
圖 6.2：生成重建之來源影像的範例。

為改善當前缺乏明確的評估指標的問題，本研究提出反轉評估技術評估動作轉換的性能。如圖 6.1所示，本系統在測試階段採用相對模式，因此要輸入來源影像(Source)、驅動影像(Driving)和初始影像(Initial)，才能輸出一張動作重演影像(Animated)。如圖 6.2所示，在已知一張動作重演影像的對應之來源影像、對應之驅動影像和對應之初始影像的情況下，若將圖 6.1的動作重演影像作為來源影像、圖 6.1的初始影像作為驅動影像、圖 6.1的驅動影像作為初始影像，即可得到一張重建之來源影像(Reconstructed)。反轉評估技術以來源影像為基準真相，藉由比較重建之來源影像和來源影像的差距，間接評估動作重演系統的性能。若兩張影像的差距愈小，則說明該動作重演系統的性能愈優秀。

## 7.對照實驗
為比較本系統和TPSMM在動畫人物動作重演性能，本研究分別讓各系統透過Animated500測試集生成500部動作重演動畫，再將所有動作重演動畫的每張影格轉換為重建之來源影像，最後比較所有重建之來源影像和來源影像的平均L1距離。若系統取得的平均L1距離愈小，則說明該系統在動畫人物動作重演的性能愈優秀。
<div align="center" style="background-color: white; border: 1px solid white; padding: 10px;">
    <img src="./docs/table1.png" alt="生成重建之來源影像的範例。"/>
</div>
表7.1：TPSMM和Lifelike在Animated500測試集實驗結果。
如表7.1所示，TPSMM的平均L1距離為11.32，而本系統的平均L1距離為10.7，本系統在指標的表現略勝於TPSMM，表明本系統比TPSMM更適合應用於動畫人物動作重演。如圖 7.1和圖 7.2所示，本研究使用相同的輸入條件，分別讓TPSMM和本系統生成一段動作重演動畫。經由轉換結果可發現TPSMM沒有確實將真實人類的眼睛和嘴巴動作轉換至動畫人物，導致動畫人物在進行眼睛和嘴巴的開合動作較為僵硬；反觀本系統則是能夠預測較為精確的動作轉換，讓動畫人物可隨著真實人類眼睛和嘴巴進行靈活的開合動作，實現靈活動作的動畫人物動作重演。

<div align="center">
    <img src="./docs/LVST1.png" alt="TPSMM和Lifelike動作重演動畫範例1。"/>
</div>
圖 7.1：TPSMM[Zha22]和Lifelike動作重演動畫範例1
<div align="center">
    <img src="./docs/LVST2.png" alt="TPSMM和Lifelike動作重演動畫範例2。"/>
</div>
圖 7.2：TPSMM[Zha22]和Lifelike動作重演動畫範例2

## 8.消融研究
