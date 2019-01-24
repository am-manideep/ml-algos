classdef Helper
    methods(Static)
        function [centroids, labels]=k_means(k, X, Iter, val)
            centroids = Helper.get_centers(k, X, val);
            labels=zeros(size(X, 1), 1);
            for i = 1:Iter
                distance = ml_sqrDist(X', centroids');
                [~, new_labels] = min(distance, [], 2);
                if isequal(labels, new_labels)
                    disp(i);
                    break;
                end
                labels = new_labels;
                for j = 1:k
                    cluster = X(labels == j, :);
                    centroids(j, :) = mean(cluster, 1);
                end
            end
        end
        
        function centers = get_centers(k, X, val)
            centers = zeros(k, size(X, 2));
            indices = (1:100);
            if val == 1
                indices = randperm(size(X, 1));
            end
            for j = 1:k
                centers(j, :) = X(indices(j), :);
            end
        end
        
        function [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma)
            trainK = zeros(size(trD, 2), size(trD, 2));
            testK = zeros(size(tstD, 2), size(trD, 2));
            for i = 1:size(trD, 2)
                for j = 1:size(trD, 2)
                    trainK(i, j) = exp((-1/gamma)*sum((trD(:, i) - trD(:, j)).^2./(trD(:, i) + trD(:, j) + eps*ones(size(trD, 1), 1))));
                end
            end
            
            for i = 1:size(tstD, 2)
                for j = 1:size(trD, 2)
                    testK(i, j) = exp((-1/gamma)*sum((tstD(:, i) - trD(:, j)).^2./(tstD(:, i) + trD(:, j) + eps*ones(size(tstD, 1), 1))));
                end
            end
        end
    end
end