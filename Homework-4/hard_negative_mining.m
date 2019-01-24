utils = HW4_Utils();
helper = SVM_Helper();
C = 10;
Iter = 10;
warning('off','all')

[trD, trLb, valD, valLb, trRegs, valRegs] = utils.getPosAndRandomNeg();
% alpha = helper.qp_func(trD, trLb, C);
% [w, b] = helper.wb_func(trD, trLb, alpha, C);
% [valPred, accuracy] = helper.ac_func(w, b, valD, valLb);
% 
% utils.genRsltFile(w, b, 'val', "val_result.mat");
% [ap, prec, rec] = utils.cmpAP("val_result.mat", 'val');

[w, b, objectives, aps] = hardNegativeMining(trD, trLb, C, Iter);

function [w, b, objectives, aps] = hardNegativeMining(trD, trLb, C, Iter)
    objectives = zeros(Iter+1, 1);
    aps = zeros(Iter+1, 1);
    [alpha, obj] = SVM_Helper.qp_func(trD, trLb, C);
    [w, b] = SVM_Helper.wb_func(trD, trLb, alpha, C);
    HW4_Utils.genRsltFile(w, b, 'val', "112028167_val.mat");
    [ap_new, ~, ~] = HW4_Utils.cmpAP("112028167_val.mat", 'val');
    objectives(1) = obj;
    disp(obj);
    aps(1) = ap_new;
    disp(ap_new)
    
    load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, 'train'), 'ubAnno');
    
    for i = 1:Iter
        fprintf('Iteration Running: %d\n',i)
        posD = [];
        negD = [];
        for j = 1:size(trLb, 1)
           if trLb(j) == 1
               posD = [posD trD(:, j)];
           else
               if alpha(j) - 0.001 > 0
                   negD = [negD trD(:, j)];
               end
           end
        end
        disp(size(negD))
        
        hne = [];
        for j = 1:93
            ubs = ubAnno{j};
            im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, 'train', j));
            img_rects = HW4_Utils.detect(im, w, b);
                
            num_positives = sum(img_rects(end, :)>0);
%             num_negatives = sum(img_rects(end, :)<0);
%             img_rects = img_rects(:,end-num_negatives+1:end);
%             if num_positives == 0
%                 disp("zero positives")
%                 img_rects = img_rects(:,num_positives+1:end);
%             else
%                 img_rects = img_rects(:,1:num_positives);
%             end
            img_rects = img_rects(:,1:num_positives+10);

            [imH, imW,~] = size(im);
            badIdxs = or(img_rects(3,:) > imW, img_rects(4,:) > imH);
            img_rects = img_rects(:,~badIdxs);
            
            % Remove random rects that overlap more than 30% with an annotated upper body
            for k=1:size(ubs,2)
                overlap = HW4_Utils.rectOverlap(img_rects, ubs(:,k));                    
                img_rects = img_rects(:, overlap < 0.1);
                if isempty(img_rects)
                    break;
                end
            end
            
%             disp(length(img_rects))
%             disp(size(img_rects))
            for k = 1:size(img_rects, 2)
                imReg = im(img_rects(2,k):img_rects(4,k), img_rects(1,k):img_rects(3,k),:);
                imReg = imresize(imReg, HW4_Utils.normImSz);
                feature = HW4_Utils.cmpFeat(rgb2gray(imReg));
                feature = feature / norm(feature);
                hne = [hne feature];
            end
            
            if size(hne, 2) > 1000
                break;
            end    
        end
        disp(size(hne))
        negD = [negD hne];
        trD = [posD negD];
        trLb = ones(size(posD, 2), 1);
        trLb = vertcat(trLb, -1*ones(size(negD, 2), 1));
        
        [alpha, obj] = SVM_Helper.qp_func(trD, trLb, C);
        [w, b] = SVM_Helper.wb_func(trD, trLb, alpha, C);
        objectives(i+1) = obj;
        disp(obj)
        
        HW4_Utils.genRsltFile(w, b, 'val', "112028167_val.mat");
        [ap_new, ~, ~] = HW4_Utils.cmpAP("112028167_val.mat", 'val');
        aps(i+1) = ap_new;
        disp(ap_new)
    end
end 

% utils.genRsltFile(w, b, 'test', "112028167.mat");