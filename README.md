# K-NN
### **目標:**
➢ 利用Iris dataset測試K-NN分類器，並求出其分類率。
### **資料描述:**
➢ 鳶尾花資料為機器學習領域中，常被用來驗算演算法優劣的資料庫。數據庫
中包含三種不同鳶尾花(山鳶尾、變色鳶尾以及維吉尼亞鳶尾)。每種花有50
筆樣本，每筆樣本以花萼長度、花萼寬度、花瓣長度以及花瓣寬度四種數值
作為特徵，進行後續定量分析。

➢ 讀取鳶尾花資料後會產生150×5的陣列，其中第5行(5th column)為
資料的類別標籤。

### **內容:**
1. 將各類別資料中的前一半資料當作測試資料(Training data)，剩下的
後一半資料當作測試資料(Testing data)，求得一個分類率；之後再將Training
data和Testing data互換，求得第二個分類率，再將兩分類率平均。
2. K-NN中之K值取1和3。
3. 利用兩兩特徵畫出散佈圖(Scatter plot)，共6張圖。
4. 利用Iris dataset測試K-NN分類器，並列出所有可能之特徵組合(共15種組合)
的分類率。
特徵組合總共15種，1為花萼長度、2為花萼寬度、3為花瓣長度、4為花瓣寬度。

| 特徵組合      | K=1(分類率)           | K=3(分類率)  |
|:-------------:|:-------------:|:-------------:|
| 1      |  0.6667     |  0.6667   |
| 2      |  0.6667     |  0.6667   |
| 3      |  0.6667     |  0.6667   |
| 4      |  0.6667     |  0.6667   |
| 12     |  0.7400      |  0.7467   |
| 13     |  0.7400      |  0.7467   |
| 14     |  0.7400      |  0.7467   |
| 23     |  0.6533      | 0.6667     |
| 24     |  0.6533      | 0.6667    |
| 34     |  0.9133      | 0.9267   |
| 123    |  0.7133      | 0.7600    |
| 134    |  0.9267      | 0.9267    |
| 124    |  0.7133      | 0.7600    |
| 234    |  0.9200      | 0.9200    |
| 1234   |  0.9267      | 0.9267   |
