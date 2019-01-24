classdef SVM_Helper
    methods(Static)
        function [alpha, objective] = qp_func(trD, trLb, C)
            X = transpose(trD) * trD;
            H = double((trLb * transpose(trLb)) .* X);
            f = -1 * ones(size(trLb,1),1);
            zero_matrix = zeros(size(trLb,1),size(trLb,1)-1);
            Aeq = transpose([trLb zero_matrix]);
            beq = zeros(size(trLb,1),1);
            lb = zeros(size(trLb,1),1);
            ub = C * ones(size(trLb,1),1);

            [alpha, objective] = quadprog(H, f, [], [], Aeq, beq, lb, ub);
            objective = -1*objective;
        end

        function [w, b] = wb_func(trD, trLb, alpha, C)
            w = trD * (trLb .* alpha);
            alpha_valid = alpha(alpha > 0.001 & C - alpha > 0.001);
            id = alpha==alpha_valid(1);
            xk = trD(:, id);
            yk = trLb(id);
            b = yk - (transpose(w) * xk);
        end

        function [valPred, accuracy] = ac_func(w, b, valD, valLb)
            valD = [valD;ones(1,size(valD,2))];
            valPred = transpose(valD) * [w;b];
            valPred(valPred > 0) = 1;
            valPred(valPred < 0) = -1;
            ac_array = valLb .* valPred;
            accuracy = (sum(ac_array(:) == 1))/size(valLb,1);
        end
        
        function objective = obj_func(trD, trLb, alpha)
            X = transpose(trD) * trD;
            H = double((trLb * transpose(trLb)) .* X);
            f = ones(size(trLb,1),1);
            
            objective = (transpose(f) * alpha) - (0.5 * (transpose(alpha) * H * alpha));
        end
        
        function negD = getNegativeExamples(w, b, dataset)
%             imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', HW4_Utils.dataDir, dataset), 'jpg');
%             nIm = length(imFiles);            
%             allRects = cell(1, nIm);
%             startT = tic;
%             for i=1:nIm
%                 ml_progressBar(i, nIm, 'Ub detection', startT);
%                 imD = imread(imFiles{i});
%                 allRects{i} = HW4_Utils.detect(imD, w, b);
%             end
            
            load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, dataset), 'ubAnno');
            [~, negD, ~, ~] = deal(cell(1, length(ubAnno)));
            for i=1:93
                ml_progressBar(i, length(ubAnno), 'Processing image');
                ubs = ubAnno{i}; % annotated upper body
%                 rects = allRects{i};
%                 num_negatives = sum(rects(end, :)<0);
%                 rects = rects(:,end-num_negatives+1:end);
                
                im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, dataset, i));
                rects = HW4_Utils.detect(im, w, b);
                
                num_negatives = sum(rects(end, :)<0);
                rects = rects(:,end-num_negatives+1:end);
                
                [imH, imW,~] = size(im);
                badIdxs = or(rects(3,:) > imW, rects(4,:) > imH);
                rects = rects(:,~badIdxs);
                
                % Remove random rects that overlap more than 30% with an annotated upper body
                for j=1:size(ubs,2)
                    overlap = HW4_Utils.rectOverlap(rects, ubs(:,j));                    
                    rects = rects(:, overlap < 0.3);
                    if isempty(rects)
                        break;
                    end
                end
                allRects{i} = rects;
                [D_i, R_i] = deal(cell(1, size(rects, 2)));
                for j=1:size(rects, 2)
                    imReg = im(rects(2,j):rects(4,j), rects(1,j):rects(3,j),:);
                    imReg = imresize(imReg, HW4_Utils.normImSz);
                    R_i{j} = imReg;
                    D_i{j} = HW4_Utils.cmpFeat(rgb2gray(imReg));                    
                end
                negD{i} = cat(2, D_i{:});
            end
            negD = cat(2, negD{:});
        end
    end
end