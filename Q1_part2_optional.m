
% Loading the data
load('HW1Dataset.mat')
% Generating indices for cross validation
cvind=crossvalind('Kfold',23040,10);
%Indexing the data
inds1=find(cvind==1);inds2=find(cvind==2);inds3=find(cvind==3);inds4=find(cvind==4);inds5=find(cvind==5);inds6=find(cvind==6);inds7=find(cvind==7);inds8=find(cvind==8);inds9=find(cvind==9);inds10=find(cvind==10);
z=[inds1 inds2 inds3 inds4 inds5 inds6 inds7 inds8 inds9 inds10];
n = 10 ;
M = cell(n, 1) ;
for i=1:n
M{i}=z(:,i);
end
% Preparing the data for training using indexes generated above and storing
% it in cell array
Train=cell(n,1);
for j=1:n
Train{j}=removerows(data,'ind',M{j});
end
% Preparing the data for testing using indexes generated above nd storing
% it in cell array
Test_data=cell(n,1);
for i=1:n
Test_data{i}=data(M{i},:);
end
%Normalizing the training & Testing data
Norm_Train=cell(10,1);
Norm_Test=cell(10,1);
for j=1:10
for i =1:8
avg1=mean(Train{j}(:,i));
dev1=std(Train{j}(:,i));
Norm_Train{j}(:,i)=((Train{j}(:,i)-avg1)/dev1);
Norm_Test{j}(:,i)=((Test_data{j}(:,i)-avg1)/dev1);
end
end
% Indexing the labels for each training and test dataset 
New_labels=double(labels);
New_labels(New_labels==1,1)=2;
New_labels(New_labels==0,1)=1;  

Train_label=cell(10,1);

for j=1:10
Train_label{j}=removerows(New_labels,'ind',M{j});
end

Test_label=cell(10,1);
for i=1:10
Test_label{i}=New_labels(M{i},:);
end

% Buiilding the Logistic Regression Model
Models=cell(10,1);
for i=1:10
Models{i}=mnrfit(Norm_Train{i},Train_label{i});
end

Train_Eval_Models=cell(10,1);
Test_Eval_Models=cell(10,1);
for i=1:10
Train_Eval_Models{i}=mnrval(Models{i},Norm_Train{i});
Test_Eval_Models{i}=mnrval(Models{i},Norm_Test{i});
end
% Changing the labels
for j=1:10
for i= 1 : 20736;
    if Train_Eval_Models{j}(i,1)>Train_Eval_Models{j}(i,2);
      Train_Eval_Models{j}(i,3)=1;
   else 
      Train_Eval_Models{j}(i,3)=2;
   end;
end;
end;


for j=1:10
for i= 1 : 2304;
    if Test_Eval_Models{j}(i,1)>Test_Eval_Models{j}(i,2);
      Test_Eval_Models{j}(i,3)=1;
   else 
      Test_Eval_Models{j}(i,3)=2;
   end;
end;
end;
% Tabulating the performance and storing in cell array
performance_train=cell(10,1);
for i=1:10
performance_train{i}=classperf(Train_label{i},Train_Eval_Models{i}(:,3));
end

performance_test=cell(10,1);
for i=1:10
performance_test{i}=classperf(Test_label{i},Test_Eval_Models{i}(:,3));
end

%Generating Train and Test confusion matrix for each dataset and storing in
%a cell array
Train_conf_matrix=cell(10,1);
for i=1:10
Train_conf_matrix{i}=confusionmat(Train_label{i},Train_Eval_Models{i}(:,3));
end

Test_conf_matrix=cell(10,1);
for i=1:10
Test_conf_matrix{i}=confusionmat(Test_label{i},Test_Eval_Models{i}(:,3));
end
%Obtaining metric results 
for i=1:10
Training_accuracy(:,i)=(Train_conf_matrix{i}(1,1)+Train_conf_matrix{i}(2,2))/(Train_conf_matrix{i}(1,1)+Train_conf_matrix{i}(2,2)+Train_conf_matrix{i}(1,2)+Train_conf_matrix{i}(2,1));
Testing_accuracy(:,i)=(Test_conf_matrix{i}(1,1)+Test_conf_matrix{i}(2,2))/(Test_conf_matrix{i}(1,1)+Test_conf_matrix{i}(2,2)+Test_conf_matrix{i}(1,2)+Test_conf_matrix{i}(2,1));
Training_senstivity(:,i)=(Train_conf_matrix{i}(2,2))/(Train_conf_matrix{i}(2,1)+Train_conf_matrix{i}(2,2));
Testing_senstivity(:,i)=(Test_conf_matrix{i}(2,2))/(Test_conf_matrix{i}(2,1)+Test_conf_matrix{i}(2,2));
Training_specificity(:,i)=(Train_conf_matrix{i}(1,1))/(Train_conf_matrix{i}(1,1)+Train_conf_matrix{i}(1,2));
Testing_specificity(:,i)=(Test_conf_matrix{i}(1,1))/(Test_conf_matrix{i}(1,1)+Test_conf_matrix{i}(1,2));
end
Final_CV_Training_Results=[mean(Training_accuracy) mean(Training_senstivity) mean(Training_specificity)]
Final_CV_Testing_Results=[mean(Testing_accuracy) mean(Testing_senstivity) mean(Testing_specificity)]

% Naive Bayes
NB_Train_label=cell(10,1);
% Preparing the data for training using indexes generated and storing
% it in cell array
for j=1:10
NB_Train_label{j}=removerows(labels,'ind',M{j});
end

NB_Test_label=cell(10,1);
for i=1:10
NB_Test_label{i}=labels(M{i},:);
end

% Building the model
NB_Models=cell(10,1);
for i=1:10
NB_Models{i}=fitcnb(Norm_Train{i},NB_Train_label{i});
%NB_Models{i}=NaiveBayes.fit(Norm_Train{i},NB_Train_label{i});
end

% Implementing the model to get training and test predictions
NB_Train_Eval_Models=cell(10,1);
NB_Test_Eval_Models=cell(10,1);
for i=1:10
NB_Train_Eval_Models{i}=predict(NB_Models{i},Norm_Train{i});
NB_Test_Eval_Models{i}=predict(NB_Models{i},Norm_Test{i});
end

% Generating confusion Matrices
NB_Train_conf_matrix=cell(10,1);
for i=1:10
NB_Train_conf_matrix{i}=confusionmat(NB_Train_label{i},NB_Train_Eval_Models{i});
end

NB_Test_conf_matrix=cell(10,1);
for i=1:10
NB_Test_conf_matrix{i}=confusionmat(NB_Test_label{i},NB_Test_Eval_Models{i});
end

% Getting results
for i=1:10
NB_Training_accuracy(:,i)=(NB_Train_conf_matrix{i}(1,1)+NB_Train_conf_matrix{i}(2,2))/(NB_Train_conf_matrix{i}(1,1)+NB_Train_conf_matrix{i}(2,2)+NB_Train_conf_matrix{i}(1,2)+NB_Train_conf_matrix{i}(2,1));
NB_Testing_accuracy(:,i)=(NB_Test_conf_matrix{i}(1,1)+NB_Test_conf_matrix{i}(2,2))/(NB_Test_conf_matrix{i}(1,1)+NB_Test_conf_matrix{i}(2,2)+NB_Test_conf_matrix{i}(1,2)+NB_Test_conf_matrix{i}(2,1));
NB_Training_senstivity(:,i)=(NB_Train_conf_matrix{i}(2,2))/(NB_Train_conf_matrix{i}(2,1)+NB_Train_conf_matrix{i}(2,2));
NB_Testing_senstivity(:,i)=(NB_Test_conf_matrix{i}(2,2))/(NB_Test_conf_matrix{i}(2,1)+NB_Test_conf_matrix{i}(2,2));
NB_Training_specificity(:,i)=(NB_Train_conf_matrix{i}(1,1))/(NB_Train_conf_matrix{i}(1,1)+NB_Train_conf_matrix{i}(1,2));
NB_Testing_specificity(:,i)=(NB_Test_conf_matrix{i}(1,1))/(NB_Test_conf_matrix{i}(1,1)+NB_Test_conf_matrix{i}(1,2));
end
% Printing the results
NB_Final_CV_Training_Results=[mean(NB_Training_accuracy) mean(NB_Training_senstivity) mean(NB_Training_specificity)]
NB_Final_CV_Testing_Results=[mean(NB_Testing_accuracy) mean(NB_Testing_senstivity) mean(NB_Testing_specificity)]


%%% Implementing LDA
LDA_Train_label=cell(10,1);
for j=1:10
LDA_Train_label{j}=removerows(labels,'ind',M{j});
end

LDA_Test_label=cell(10,1);
for i=1:10
LDA_Test_label{i}=labels(M{i},:);
end
%Building Model
%LDA_Models=cell(10,1);
%for i=1:10
%LDA_Models{i}=fitcdiscr(Norm_Train{i},LDA_Train_label{i});
%end
% Applying the model
%LDA_Train_Eval_Models=cell(10,1);
%LDA_Test_Eval_Models=cell(10,1);
%for i=1:10
%LDA_Train_Eval_Models{i}=predict(LDA_Models{i},Norm_Train{i});
%LDA_Test_Eval_Models{i}=predict(LDA_Models{i},Norm_Test{i});
%end

LDA_Train_Eval_Models=cell(10,1);
LDA_Test_Eval_Models=cell(10,1);
for i=1:10
LDA_Train_Eval_Models{i}=classify(Norm_Train{i},Norm_Train{i},LDA_Train_label{i});
LDA_Test_Eval_Models{i}=classify(Norm_Test{i},Norm_Train{i},LDA_Train_label{i});
end

% Generating confusion matrix
LDA_Train_conf_matrix=cell(10,1);
for i=1:10
LDA_Train_conf_matrix{i}=confusionmat(LDA_Train_label{i},LDA_Train_Eval_Models{i});
end

LDA_Test_conf_matrix=cell(10,1);
for i=1:10
LDA_Test_conf_matrix{i}=confusionmat(LDA_Test_label{i},LDA_Test_Eval_Models{i});
end

% Getting results
for i=1:10
LDA_Training_accuracy(:,i)=(LDA_Train_conf_matrix{i}(1,1)+LDA_Train_conf_matrix{i}(2,2))/(LDA_Train_conf_matrix{i}(1,1)+LDA_Train_conf_matrix{i}(2,2)+LDA_Train_conf_matrix{i}(1,2)+LDA_Train_conf_matrix{i}(2,1));
LDA_Testing_accuracy(:,i)=(LDA_Test_conf_matrix{i}(1,1)+LDA_Test_conf_matrix{i}(2,2))/(LDA_Test_conf_matrix{i}(1,1)+LDA_Test_conf_matrix{i}(2,2)+LDA_Test_conf_matrix{i}(1,2)+LDA_Test_conf_matrix{i}(2,1));
LDA_Training_senstivity(:,i)=(LDA_Train_conf_matrix{i}(2,2))/(LDA_Train_conf_matrix{i}(2,1)+LDA_Train_conf_matrix{i}(2,2));
LDA_Testing_senstivity(:,i)=(LDA_Test_conf_matrix{i}(2,2))/(LDA_Test_conf_matrix{i}(2,1)+LDA_Test_conf_matrix{i}(2,2));
LDA_Training_specificity(:,i)=(LDA_Train_conf_matrix{i}(1,1))/(LDA_Train_conf_matrix{i}(1,1)+LDA_Train_conf_matrix{i}(1,2));
LDA_Testing_specificity(:,i)=(LDA_Test_conf_matrix{i}(1,1))/(LDA_Test_conf_matrix{i}(1,1)+LDA_Test_conf_matrix{i}(1,2));
end

LDA_Final_CV_Training_Results=[mean(LDA_Training_accuracy) mean(LDA_Training_senstivity) mean(LDA_Training_specificity)]
LDA_Final_CV_Testing_Results=[mean(LDA_Testing_accuracy) mean(LDA_Testing_senstivity) mean(LDA_Testing_specificity)]

% LINEAR SVM 
SVM_Train_label=cell(10,1);

for j=1:10
SVM_Train_label{j}=removerows(labels,'ind',M{j});
end

SVM_Test_label=cell(10,1);
for i=1:10
SVM_Test_label{i}=labels(M{i},:);
end

% Setting up the model
smoopt = statset('Maxiter',50000);

SVM_Models=cell(10,1);
for i=1:10
SVM_Models{i}=svmtrain(Norm_Train{i},SVM_Train_label{i},'autoscale','false','Method','SMO','SMO_Opts',smoopt);    
%SVM_Models{i}=fitcsvm(Norm_Train{i},SVM_Train_label{i},'Standardize',false,'Solver','SMO','IterationLimit',50000);
end
% Testing the model
SVM_Train_Eval_Models=cell(10,1);
SVM_Test_Eval_Models=cell(10,1);
for i=1:10
SVM_Train_Eval_Models{i}= svmclassify(SVM_Models{i},Norm_Train{i});
SVM_Test_Eval_Models{i}= svmclassify(SVM_Models{i},Norm_Test{i});
end
% Generating confusion matrix
SVM_Train_conf_matrix=cell(10,1);
for i=1:10
SVM_Train_conf_matrix{i}=confusionmat(SVM_Train_label{i},SVM_Train_Eval_Models{i});
end

SVM_Test_conf_matrix=cell(10,1);
for i=1:10
SVM_Test_conf_matrix{i}=confusionmat(SVM_Test_label{i},SVM_Test_Eval_Models{i});
end

% Getting results
for i=1:10
SVM_Training_accuracy(:,i)=(SVM_Train_conf_matrix{i}(1,1)+SVM_Train_conf_matrix{i}(2,2))/(SVM_Train_conf_matrix{i}(1,1)+SVM_Train_conf_matrix{i}(2,2)+SVM_Train_conf_matrix{i}(1,2)+SVM_Train_conf_matrix{i}(2,1));
SVM_Testing_accuracy(:,i)=(SVM_Test_conf_matrix{i}(1,1)+SVM_Test_conf_matrix{i}(2,2))/(SVM_Test_conf_matrix{i}(1,1)+SVM_Test_conf_matrix{i}(2,2)+SVM_Test_conf_matrix{i}(1,2)+SVM_Test_conf_matrix{i}(2,1));
SVM_Training_senstivity(:,i)=(SVM_Train_conf_matrix{i}(2,2))/(SVM_Train_conf_matrix{i}(2,1)+SVM_Train_conf_matrix{i}(2,2));
SVM_Testing_senstivity(:,i)=(SVM_Test_conf_matrix{i}(2,2))/(SVM_Test_conf_matrix{i}(2,1)+SVM_Test_conf_matrix{i}(2,2));
SVM_Training_specificity(:,i)=(SVM_Train_conf_matrix{i}(1,1))/(SVM_Train_conf_matrix{i}(1,1)+SVM_Train_conf_matrix{i}(1,2));
SVM_Testing_specificity(:,i)=(SVM_Test_conf_matrix{i}(1,1))/(SVM_Test_conf_matrix{i}(1,1)+SVM_Test_conf_matrix{i}(1,2));
end


SVM_Final_CV_Training_Results=[mean(SVM_Training_accuracy) mean(SVM_Training_senstivity) mean(SVM_Training_specificity)]
SVM_Final_CV_Testing_Results=[mean(SVM_Testing_accuracy) mean(SVM_Testing_senstivity) mean(SVM_Testing_specificity)]

% RBF SVM
RBF_SVM_Train_label=cell(10,1);

for j=1:10
RBF_SVM_Train_label{j}=removerows(labels,'ind',M{j});
end

RBF_SVM_Test_label=cell(10,1);
for i=1:10
RBF_SVM_Test_label{i}=labels(M{i},:);
end
% Setting up model
smoopt = statset('Maxiter',50000);

RBF_SVM_Models=cell(10,1);
for i=1:10
RBF_SVM_Models{i}=fitcsvm(Norm_Train{i},RBF_SVM_Train_label{i},'Standardize',false,'Solver','SMO','KernelFunction','rbf','IterationLimit',50000);
end
% Applying the model
RBF_SVM_Train_Eval_Models=cell(10,1);
RBF_SVM_Test_Eval_Models=cell(10,1);
for i=1:10
RBF_SVM_Train_Eval_Models{i}=predict(RBF_SVM_Models{i},Norm_Train{i});
RBF_SVM_Test_Eval_Models{i}=predict(RBF_SVM_Models{i},Norm_Test{i});
end
% Obtaining confusion matrix
RBF_SVM_Train_conf_matrix=cell(10,1);
for i=1:10
RBF_SVM_Train_conf_matrix{i}=confusionmat(RBF_SVM_Train_label{i},RBF_SVM_Train_Eval_Models{i});
end

RBF_SVM_Test_conf_matrix=cell(10,1);
for i=1:10
RBF_SVM_Test_conf_matrix{i}=confusionmat(RBF_SVM_Test_label{i},RBF_SVM_Test_Eval_Models{i});
end

%Getting the results
for i=1:10
RBF_SVM_Training_accuracy(:,i)=(RBF_SVM_Train_conf_matrix{i}(1,1)+RBF_SVM_Train_conf_matrix{i}(2,2))/(RBF_SVM_Train_conf_matrix{i}(1,1)+RBF_SVM_Train_conf_matrix{i}(2,2)+RBF_SVM_Train_conf_matrix{i}(1,2)+RBF_SVM_Train_conf_matrix{i}(2,1));
RBF_SVM_Testing_accuracy(:,i)=(RBF_SVM_Test_conf_matrix{i}(1,1)+RBF_SVM_Test_conf_matrix{i}(2,2))/(RBF_SVM_Test_conf_matrix{i}(1,1)+RBF_SVM_Test_conf_matrix{i}(2,2)+RBF_SVM_Test_conf_matrix{i}(1,2)+RBF_SVM_Test_conf_matrix{i}(2,1));
RBF_SVM_Training_senstivity(:,i)=(RBF_SVM_Train_conf_matrix{i}(2,2))/(RBF_SVM_Train_conf_matrix{i}(2,1)+RBF_SVM_Train_conf_matrix{i}(2,2));
RBF_SVM_Testing_senstivity(:,i)=(RBF_SVM_Test_conf_matrix{i}(2,2))/(RBF_SVM_Test_conf_matrix{i}(2,1)+RBF_SVM_Test_conf_matrix{i}(2,2));
RBF_SVM_Training_specificity(:,i)=(RBF_SVM_Train_conf_matrix{i}(1,1))/(RBF_SVM_Train_conf_matrix{i}(1,1)+RBF_SVM_Train_conf_matrix{i}(1,2));
RBF_SVM_Testing_specificity(:,i)=(RBF_SVM_Test_conf_matrix{i}(1,1))/(RBF_SVM_Test_conf_matrix{i}(1,1)+RBF_SVM_Test_conf_matrix{i}(1,2));
end


RBF_SVM_Final_CV_Training_Results=[mean(RBF_SVM_Training_accuracy) mean(RBF_SVM_Training_senstivity) mean(RBF_SVM_Training_specificity)]
RBF_SVM_Final_CV_Testing_Results=[mean(RBF_SVM_Testing_accuracy) mean(RBF_SVM_Testing_senstivity) mean(RBF_SVM_Testing_specificity)]

% ANN Normalized
ANN_Train_label=cell(10,1);
for j=1:10
ANN_Train_label{j}=removerows(labels,'ind',M{j});
end

ANN_Test_label=cell(10,1);
for i=1:10
ANN_Test_label{i}=labels(M{i},:);
end
% Builiding the Model
ANN_Models=cell(10,1);
for i=1:10
ANN_Models{i}=newff(minmax(Norm_Train{i}'),[3 1], {'tansig' 'purelin'});
ANN_Models{i}.trainParam.epochs=200;
ANN_Models{i} = train(ANN_Models{i}, Norm_Train{i}', ANN_Train_label{i}');
end

ANN_Train_Eval_Models=cell(10,1);
ANN_Test_Eval_Models=cell(10,1);
for i=1:10
ANN_Train_Eval_Models{i}=round(sim(ANN_Models{i},Norm_Train{i}'));
ANN_Test_Eval_Models{i}=round(sim(ANN_Models{i},Norm_Test{i}'));
end
% Getting confusion matrix
ANN_Train_conf_matrix=cell(10,1);
for i=1:10
ANN_Train_conf_matrix{i}=confusionmat(ANN_Train_label{i},logical(ANN_Train_Eval_Models{i}'));
end

ANN_Test_conf_matrix=cell(10,1);
for i=1:10
ANN_Test_conf_matrix{i}=confusionmat(ANN_Test_label{i},logical(ANN_Test_Eval_Models{i}'));
end

% Getting results
for i=1:10
ANN_Training_accuracy(:,i)=(ANN_Train_conf_matrix{i}(1,1)+ANN_Train_conf_matrix{i}(2,2))/(ANN_Train_conf_matrix{i}(1,1)+ANN_Train_conf_matrix{i}(2,2)+ANN_Train_conf_matrix{i}(1,2)+ANN_Train_conf_matrix{i}(2,1));
ANN_Testing_accuracy(:,i)=(ANN_Test_conf_matrix{i}(1,1)+ANN_Test_conf_matrix{i}(2,2))/(ANN_Test_conf_matrix{i}(1,1)+ANN_Test_conf_matrix{i}(2,2)+ANN_Test_conf_matrix{i}(1,2)+ANN_Test_conf_matrix{i}(2,1));
ANN_Training_senstivity(:,i)=(ANN_Train_conf_matrix{i}(2,2))/(ANN_Train_conf_matrix{i}(2,1)+ANN_Train_conf_matrix{i}(2,2));
ANN_Testing_senstivity(:,i)=(ANN_Test_conf_matrix{i}(2,2))/(ANN_Test_conf_matrix{i}(2,1)+ANN_Test_conf_matrix{i}(2,2));
ANN_Training_specificity(:,i)=(ANN_Train_conf_matrix{i}(1,1))/(ANN_Train_conf_matrix{i}(1,1)+ANN_Train_conf_matrix{i}(1,2));
ANN_Testing_specificity(:,i)=(ANN_Test_conf_matrix{i}(1,1))/(ANN_Test_conf_matrix{i}(1,1)+ANN_Test_conf_matrix{i}(1,2));
end


ANN_Final_CV_Training_Results=[mean(ANN_Training_accuracy) mean(ANN_Training_senstivity) mean(ANN_Training_specificity)]
ANN_Final_CV_Testing_Results=[mean(ANN_Testing_accuracy) mean(ANN_Testing_senstivity) mean(ANN_Testing_specificity)]

% ANN NON_NORMALIZED
N_ANN_Train_label=cell(10,1);
for j=1:10
N_ANN_Train_label{j}=removerows(labels,'ind',M{j});
end

N_ANN_Test_label=cell(10,1);
for i=1:10
N_ANN_Test_label{i}=labels(M{i},:);
end

%Building the model
N_ANN_Models=cell(10,1);
for i=1:10
N_ANN_Models{i}=newff(minmax(Train{i}'),[3 1], {'tansig' 'purelin'});
N_ANN_Models{i}.trainParam.epochs=200;
N_ANN_Models{i} = train(N_ANN_Models{i}, Train{i}', N_ANN_Train_label{i}');
end



N_ANN_Train_Eval_Models=cell(10,1);
N_ANN_Test_Eval_Models=cell(10,1);
for i=1:10
N_ANN_Train_Eval_Models{i}=round(sim(N_ANN_Models{i},Train{i}'));
N_ANN_Test_Eval_Models{i}=round(sim(N_ANN_Models{i},Test_data{i}'));
end
% Getting Confusion Matrix
N_ANN_Train_conf_matrix=cell(10,1);
for i=1:10
N_ANN_Train_conf_matrix{i}=confusionmat(N_ANN_Train_label{i},logical(N_ANN_Train_Eval_Models{i}'));
end

N_ANN_Test_conf_matrix=cell(10,1);
for i=1:10
N_ANN_Test_conf_matrix{i}=confusionmat(N_ANN_Test_label{i},logical(N_ANN_Test_Eval_Models{i}'));
end

% Getting results
for i=1:10
N_ANN_Training_accuracy(:,i)=(N_ANN_Train_conf_matrix{i}(1,1)+N_ANN_Train_conf_matrix{i}(2,2))/(N_ANN_Train_conf_matrix{i}(1,1)+N_ANN_Train_conf_matrix{i}(2,2)+N_ANN_Train_conf_matrix{i}(1,2)+N_ANN_Train_conf_matrix{i}(2,1));
N_ANN_Testing_accuracy(:,i)=(N_ANN_Test_conf_matrix{i}(1,1)+N_ANN_Test_conf_matrix{i}(2,2))/(N_ANN_Test_conf_matrix{i}(1,1)+N_ANN_Test_conf_matrix{i}(2,2)+N_ANN_Test_conf_matrix{i}(1,2)+N_ANN_Test_conf_matrix{i}(2,1));
N_ANN_Training_senstivity(:,i)=(N_ANN_Train_conf_matrix{i}(2,2))/(N_ANN_Train_conf_matrix{i}(2,1)+N_ANN_Train_conf_matrix{i}(2,2));
N_ANN_Testing_senstivity(:,i)=(N_ANN_Test_conf_matrix{i}(2,2))/(N_ANN_Test_conf_matrix{i}(2,1)+N_ANN_Test_conf_matrix{i}(2,2));
N_ANN_Training_specificity(:,i)=(N_ANN_Train_conf_matrix{i}(1,1))/(N_ANN_Train_conf_matrix{i}(1,1)+N_ANN_Train_conf_matrix{i}(1,2));
N_ANN_Testing_specificity(:,i)=(N_ANN_Test_conf_matrix{i}(1,1))/(N_ANN_Test_conf_matrix{i}(1,1)+N_ANN_Test_conf_matrix{i}(1,2));
end


N_ANN_Final_CV_Training_Results=[mean(N_ANN_Training_accuracy) mean(N_ANN_Training_senstivity) mean(N_ANN_Training_specificity)]
N_ANN_Final_CV_Testing_Results=[mean(N_ANN_Testing_accuracy) mean(N_ANN_Testing_senstivity) mean(N_ANN_Testing_specificity)]