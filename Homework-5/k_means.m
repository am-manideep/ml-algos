X = load('../../hw5data/digit/digit.txt');
Y = load('../../hw5data/digit/labels.txt');

k_array = [2 4 6]';
Iter = 20;
labels = zeros(size(X, 1), size(k_array, 1));
ss = zeros(size(k_array, 1), 1);
p = zeros(size(k_array, 1), 3);

for i = 1:size(k_array, 1)
    k = k_array(i);
    [centroids, labels(:, i)] = Helper.k_means(k, X, Iter, 0);
    distance = ml_sqrDist(X', centroids');
    [dist, ~] = min(distance, [], 2);
    ss(i) = sum(dist);
    p1_total = 0;
    p2_total = 0;
    p1 = 0;
    p2 = 0;
    for m = 1:size(Y, 1) - 1
        for n = m+1:size(Y, 1)
            if m ~= n
                if Y(m) == Y(n)
                    p1_total = p1_total + 1;
                else
                    p2_total = p2_total + 1;
                end
                if Y(m) == Y(n) && labels(m, i) == labels(n, i)
                    p1 = p1 + 1;
                end
                if Y(m) ~= Y(n) && labels(m, i) ~= labels(n, i)
                    p2 = p2 + 1;
                end
            end
        end
    end
    p1 = p1 * 100 / p1_total;
    p2 = p2 * 100 /p2_total;
    p3 = (p1 + p2)/2;
    p(i, :) = [p1, p2, p3];
end

k_array = [1 2 3 4 5 6 7 8 9 10]';
Iter = 20;
labels_2 = zeros(size(X, 1), size(k_array, 1));
ss_2 = zeros(size(k_array, 1), 1);
p_2 = zeros(size(k_array, 1), 3);

for i = 1:size(k_array, 1)
    for j = 1:10
        k = k_array(i);
        [centroids, labels_2(:, i)] = Helper.k_means(k, X, Iter, 1);
        distance = ml_sqrDist(X', centroids');
        [dist, ~] = min(distance, [], 2);
        ss_2(i) = ss_2(i) + sum(dist);
        p1_total = 0;
        p2_total = 0;
        p1 = 0;
        p2 = 0;
        for m = 1:size(Y, 1) - 1
            for n = m + 1:size(Y, 1)
                if Y(m) == Y(n)
                    p1_total = p1_total + 1;
                else
                    p2_total = p2_total + 1;
                end
                if Y(m) == Y(n) && labels_2(m, i) == labels_2(n, i)
                    p1 = p1 + 1;
                end
                if Y(m) ~= Y(n) && labels_2(m, i) ~= labels_2(n, i)
                    p2 = p2 + 1;
                end
            end
        end
        p1 = p1 * 100 / p1_total;
        p2 = p2 * 100 /p2_total;
        p3 = (p1 + p2)/2;
        p_2(i, :) = p_2(i, :) + [p1, p2, p3];
    end
end
ss_2 = ss_2/10;
p_2 = p_2/10;