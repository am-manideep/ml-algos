C = 0.1;
alpha = qp_func_mc(trD, trLb, C);
[w, b] = wb_func_mc(trD, trLb, alpha, C);
[valPred, accuracy] = ac_func_mc(w, b, valD, valLb);
pred = pred_func_mc(w, b, tstD);
final_pred = zeros(size(pred, 1), 2);
for i = 1:size(pred, 1)
    final_pred(i, :) = [i pred(i)];
end
csvwrite("submission.csv", final_pred);

function alpha = qp_func_mc(trD, trLb, C)
    trLbi = zeros(size(trLb,1),1);
    alpha = zeros(size(trLb,1),10);
    for i = 1:10
        trLbi(trLb ~= i) = -1;
        trLbi(trLb == i) = 1;
        X = transpose(trD) * trD;
        H = double((trLbi * transpose(trLbi)) .* X);
        f = -1 * ones(size(trLbi,1),1);
        zero_matrix = zeros(size(trLbi,1),size(trLbi,1)-1);
        Aeq = transpose([trLbi zero_matrix]);
        beq = zeros(size(trLbi,1),1);
        lb = zeros(size(trLbi,1),1);
        ub = C * ones(size(trLbi,1),1);
        alpha(:, i) = quadprog(H, f, [], [], Aeq, beq, lb, ub);
    end
end

function [w, b] = wb_func_mc(trD, trLb, alpha, C)
    trLbi = zeros(size(trLb,1),1);
    b = zeros(10, 1);
    w = zeros(size(trD,1),10);
    for i = 1:10
        trLbi(trLb ~= i) = -1;
        trLbi(trLb == i) = 1;
        alpha_temp = alpha(:, i);
        w(:, i) = trD * (trLbi .* alpha_temp);
        id = 0;
        max_value = 0;
        for j = 1:size(alpha_temp, 1)
            if alpha_temp(j) > 0.001 && C - alpha_temp(j) > 0.001 && alpha_temp(j) > max_value
                id = j;
                max_value = alpha_temp(j);
            end
        end
%         alpha_valid = alpha_temp(alpha_temp > 0.001 & C - alpha_temp > 0.001);
%         id = alpha_temp==alpha_valid(1);
        disp(id)
        disp(alpha_temp(id))
        xk = trD(:, id);
%         disp(xk)
        yk = trLbi(id);
%         disp(yk)
%         disp(yk - (transpose(w(:, i)) * xk))
        b(i) = yk - (transpose(w(:, i)) * xk);
%         disp(b(i))
    end
end

function [valPrediction, accuracy] = ac_func_mc(w, b, valD, valLb)
    valD = [valD;ones(1,size(valD,2))];
    valPred = zeros(size(valLb,1),10);
    valPrediction = zeros(size(valLb,1),1);
    for i = 1:10
        valPred(:, i) = transpose(valD) * [w(:, i);b(i)];
    end
    [~,loc] = max(transpose(valPred));
    for i = 1:size(valLb,1)
        valPrediction(i) = loc(i);
    end
    ac_array = valLb - valPrediction;
    accuracy = (sum(ac_array(:) == 0))/size(valLb,1);
end

function valPrediction = pred_func_mc(w, b, tstD)
    tstD = [tstD;ones(1,size(tstD,2))];
    valPred = zeros(size(tstD,2),10);
    valPrediction = zeros(size(tstD,2),1);
    for i = 1:10
        valPred(:, i) = transpose(tstD) * [w(:, i);b(i)];
    end
    [~,loc] = max(transpose(valPred));
    for i = 1:size(tstD,2)
        valPrediction(i) = loc(i);
    end
end