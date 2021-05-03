clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure

%% 讀取.txt資料
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % 原始資料，75筆資料 x 4個特徵
label   = dataSet(:,5);      % 75筆資料所對應的標籤
kk = 3;
error = 0;
acc=0;

%% 範例一、Scatter Plot
figure; % 開啟新的繪圖空間

plot(rawData(  1: 50,1),rawData(  1: 50,2),'ro',...
     rawData( 51:100,1),rawData( 51:100,2),'go',...
     rawData(101:150,1),rawData(101:150,2),'bo');   
     % 以plot繪圖指令分別畫出class1~3之第一與第二特徵。

title('Scatter Plot');                              % 圖名稱
legend('class1', 'class2', 'class3');               % 類別標號說明
xlabel('Feature1');                                 % 特徵標號註解
ylabel('Feature2');                                 % 特徵標號註解

%% 範例二、計算距離
trainset = [rawData(  1: 25,1:4);...
          rawData( 51: 75,1:4);...
          rawData(101:125,1:4);]; 
          % 選取每類別前半，合併為training set

testset = [rawData( 26: 50,1:4);...
          rawData( 76:100,1:4);...
          rawData(126:150,1:4)]; 
          % 選取每類別後半，合併為test set
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

distancev=zeros(trainm,1);%每個測試點與訓練及的歐式距離量
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:trainn
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.歐式距離
    end
    [val,index] = sort(distancev,'ascend');
    
    
    M = mode(label(index(1:kk)));
    

    
   
    if M ~= label(i,end)
        error=error+1;
    end

end

CR=1-error/testm;
disp('CR:');
disp(CR);










