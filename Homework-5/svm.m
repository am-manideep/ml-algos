bow = HW5_BoW();
helper = Helper();

scales = [8, 16, 32, 64];
normH = 16;
normW = 16;
bowCs = bow.learnDictionary(scales, normH, normW);

[trIds, trLbs] = ml_load('../bigbangtheory/train.mat',  'imIds', 'lbs');             
tstIds = ml_load('../bigbangtheory/test.mat', 'imIds'); 

trD  = bow.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
tstD = bow.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);

model = svmtrain(trLbs, trD', '-v 5');
disp(model)

C_array = zeros(12, 1);
gamma_array = zeros(12, 1);
models = zeros(12, 12);

for i = 1:12
    C_array(i) = 2^(-7 + 2 * i);
    gamma_array(i) = 2^(-17 + 2 * i);
end

for i = 1:12
    for j = 1:12
        accuracy = svmtrain(trLbs, trD', sprintf('-v 5 -g %f -c %f', gamma_array(j), C_array(i)));
        models(i, j) = accuracy;
        fprintf('i = %f, j = %f, accuracy = %f', i, j, accuracy);
    end
end

C_array = zeros(12, 1);
gamma_array = zeros(12, 1);
models_2 = zeros(12, 12);

for i = 1:12
    C_array(i) = 2^(-7 + 2 * i);
    gamma_array(i) = 2^(-17 + 2 * i);
end

for i = 1:12
    for j = 1:12
        [trainK, testK] = helper.cmpExpX2Kernel(trD, tstD, gamma_array(j));
        model = svmtrain(trLbs, [(1:1777)', trainK], sprintf('-c %f -t 4 -v 5', C_array(i)));
        models_2(i, j) = model;
        fprintf('i = %f, j = %f, accuracy = %f', i, j, model);
    end
end


gamma = 2;

[trainK, testK] = helper.cmpExpX2Kernel(trD, tstD, gamma);
model = svmtrain(trLbs, [(1:1777)', trainK], sprintf('-c %f -t 4', 2^5));
[predict_label, accuracy, dec_values] = svmpredict(ones(1600, 1), [(1:1600)', testK], model);
% [predict_label, accuracy, dec_values] = svmpredict(trLbs, [(1:1777)', trainK], model);

final_pred = zeros(size(predict_label, 1), 2);
for i = 1:size(predict_label, 1)
    final_pred(i, :) = [i+1777 predict_label(i)];
end
csvwrite("submission5.csv", final_pred);