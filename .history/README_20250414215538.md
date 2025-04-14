#栩栩如生：動畫人物動作重演
# 1.簡介

# 2.系統流程
<div align="center">
    <img src="./docs/Lifelike.png" alt="Lifelike的系統流程圖。" width="70%"/>
</div>
圖 2.1：Lifelike的系統流程圖。

圖 2.1為本系統流程圖，關鍵點檢測器(Keypoint Detector)會提取10組關鍵點集合，再根據每組關鍵點集合計算1個TPS轉換。同時，背景預測器(BG Motion Predictor)會估計1個表示背景運動的仿射轉換。密集動作網路(Dense Motion Network)根據11個轉換的資訊估計多解析度光流(Multi-resolution Optical flows)和多解析度遮擋遮罩(Multi-resolution Occlusion Masks)。修復網路(Inpainting Network)會將來源影像編碼為不同尺度的特徵圖，先使用多解析度光流扭曲相應解析度的特徵圖，再使用多解析度遮擋遮罩修復相應解析度的被扭曲之特徵圖，最後輸出一張動作重演的影像。