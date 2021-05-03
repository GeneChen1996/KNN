%% Artificial Intelligence Homework#1 Demo - 2020/03/17

%%
clc;       % �M��command window
clear      % �M��workspace
close all  % �����Ҧ�figure

%% Ū��.txt���
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % ��l��ơA75����� x 4�ӯS�x
label   = dataSet(:,5);      % 75����Ʃҹ���������
kk = 3;
error = 0;
acc=0;

%% �d�Ҥ@�BScatter Plot
figure; % �}�ҷs��ø�ϪŶ�

plot(rawData(  1: 50,1),rawData(  1: 50,2),'ro',...
     rawData( 51:100,1),rawData( 51:100,2),'go',...
     rawData(101:150,1),rawData(101:150,2),'bo');   
     % �Hplotø�ϫ��O���O�e�Xclass1~3���Ĥ@�P�ĤG�S�x�C

title('Scatter Plot');                              % �ϦW��
legend('class1', 'class2', 'class3');               % ���O�и�����
xlabel('Feature1');                                 % �S�x�и�����
ylabel('Feature2');                                 % �S�x�и�����

%% �d�ҤG�B�p��Z��
trainset = [rawData(  1: 25,1:4);...
          rawData( 51: 75,1:4);...
          rawData(101:125,1:4);]; 
          % ����C���O�e�b�A�X�֬�training set

testset = [rawData( 26: 50,1:4);...
          rawData( 76:100,1:4);...
          rawData(126:150,1:4)]; 
          % ����C���O��b�A�X�֬�test set
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

distancev=zeros(trainm,1);%�C�Ӵ����I�P�V�m�Ϊ��ڦ��Z���q
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:trainn
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.�ڦ��Z��
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



% %% �d�ҤT�Bk-NN��Ϋ��O�Ѧ�
% %%% === sort: �Ƨ� =====================================================%%%
% A = [9 0 -7 5 3 8 -10 4 2];
% [value,index] = sort(A,'ascend'); 
% % �N�}�CA�A�Ѥp��j���s�Ƨ�
% % value: �Ѥp��j���s�ƧǪ���
% % index: value���C�ӭȦbA����l��m
% %%% ====================================================================%%%
% 
% %%% === mode: ���� =====================================================%%%
% B = [3 1 3 3 2];
% M = mode(B); 
% % ��X�}�CB�����W�c�X�{���ƭ�
% %%% ====================================================================%%%









