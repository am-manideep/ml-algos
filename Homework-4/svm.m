helper = SVM_Helper();

C = 10;
[alpha, obj] = helper.qp_func(trD, trLb, C);
[w, b] = helper.wb_func(trD, trLb, alpha, C);
[valPred, accuracy] = helper.ac_func(w, b, valD, valLb);
% obj = helper.obj_func(trD, trLb, alpha);
num_support_vectors = size(alpha(alpha > 0.001 & C - alpha > 0.001), 1);
confusion_matrix = confusionmat(valPred, valLb);