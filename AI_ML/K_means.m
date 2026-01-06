% Generar datos de ejemplo
rng(1); % Para reproducibilidad (Random number generator)
data = [randn(50,2) + 1; randn(50,2) - 1; randn(50,2) + [5 -5]]; % Tres grupos

% Número de clústeres
k = 3;

% Inicializar centroides aleatorios
numPoints = size(data, 1);  
centroids = data(randperm(numPoints, k), :); % randperm makes a random permutation of the dataset for take the centroids

% Inicializar variables
maxIter = 100; % Número máximo de iteraciones
prevCentroids = centroids;
labels = zeros(numPoints, 1);

for iter = 1:maxIter
    % Paso 1: Asignar cada punto al clúster más cercano
    for i = 1:numPoints
        distances = sum((data(i, :) - centroids).^2, 2); % Distancia euclidiana
        [~, labels(i)] = min(distances); % Etiqueta del clúster más cercano
    end
    
    % Paso 2: Actualizar los centroides
    for j = 1:k
        centroids(j, :) = mean(data(labels == j, :), 1); % Promedio de puntos en el clúster
    end
    
    % Verificar convergencia
    if all(prevCentroids == centroids)
        fprintf('Convergencia alcanzada en la iteración %d.\n', iter);
        break;
    end
    prevCentroids = centroids;
end

% Visualización
figure;
gscatter(data(:,1), data(:,2), labels);
hold on;
plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('k-means algorithm');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids');
hold off;
