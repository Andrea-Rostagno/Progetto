%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTE 1 - IMPLEMENTAZIONE DELL'ALGORITMO NELDER-MEAD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_min, f_min, iter, min_history, e_rate] = nelder_mead(f, x0, tol, max_iter)

    n = length(x0); % Dimension of the problem
    rho = 1;      % Reflection coefficient
    chi = 2;      % Expansion coefficient
    gamma = 0.5;      % Contraction coefficient
    sigma = 0.5;    % Shrinkage coefficient
    
    % Initialize simplex
    simplex = zeros(n + 1, n);
    simplex(1, :) = x0';
    e_rate = zeros(n,4);
    identity_matrix = eye(n); 

    for i = 2:n+1
        simplex(i, :) = x0' + 0.03 * identity_matrix(:, i-1)'; % create new points similar to the first one
    end

    % Evaluate function at simplex points
    f_vals = arrayfun(@(i) f(simplex(i, :)'), 1:n+1); % Compute the correspondent value of the function for each row (the same of sapply() in R) 
    vectorNorm = zeros(n, 1);
    min_history = zeros(1, max_iter);
    iter = 1;

    while iter < max_iter

        % Sort vertices by function values
        [f_vals, idx] = sort(f_vals);
        simplex = simplex(idx, :); 
        min_history(iter) = f_vals(1); 

        % Compute centroid of all points except worst
        centroid = mean(simplex(1:n, :)); 

        % Reflection
        x_r = centroid + rho * (centroid - simplex(n+1, :));
        f_r = f(x_r');

        if f_r < f_vals(1) % Expansion
            x_e = centroid + chi * (x_r - centroid);
            f_e = f(x_e');
            if f_e < f_r
                simplex(n+1, :) = x_e;
                f_vals(n+1) = f_e;
            else
                simplex(n+1, :) = x_r;
                f_vals(n+1) = f_r;
            end
        elseif f_r < f_vals(n) % Accept reflection
            simplex(n+1, :) = x_r;
            f_vals(n+1) = f_r;
        else % Contraction
            if f_vals(n+1) < f_r
                x_c = centroid - gamma * (centroid - f_vals(n+1));
            else
                x_c = centroid - gamma * (centroid - f_r);
            end
            f_c = f(x_c');
            if f_c < f_vals(n+1)
                simplex(n+1, :) = x_c;
                f_vals(n+1) = f_c;
            else % Shrink
                for i = 2:n+1
                    simplex(i, :) = simplex(1, :) + sigma * (simplex(i, :) - simplex(1, :));
                    f_vals(i) = f(simplex(i, :)');
                end
            end
        end

        [f_vals, idx] = sort(f_vals); 
        simplex = simplex(idx, :); 
        term_f = abs(f_vals(n+1) - f_vals(1)); 
        
        for i = 2:n+1
            vectorNorm(i-1) = norm(simplex(i,:) - simplex(1,:), inf);
        end

        term_x = max(vectorNorm);

        if iter < 5 && norm(e_rate(:,iter) - simplex(1,:)', 2) >= tol
            e_rate(:,iter) = simplex(1,:)';
        end

        if iter >= 5 && norm(e_rate(:,4) - simplex(1,:)', 2) >= tol
            e_rate = [e_rate(:, 2:4), simplex(1,:)'];
        end


        % Check convergence
        if term_f <= tol || term_x <= tol 
            break;
        end

        iter = iter + 1;
    end

    x_min = simplex(1, :);
    f_min = f_vals(1);
    min_history = min_history(1:iter);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTE 2 - TEST DELL'ALGORITMO SULLA FUNZIONE DI ROSENBROCK
% Test con due punti iniziali richiesti dall'assignment:
% [1.2, 1.2] e [-1.2, 1.0]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Nelder-Mead test on Rosenbrock function
clc; clear; close all;

% Define Rosenbrock function
rosenbrock = @(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;

% Initial points
x0_1 = [1.2, 1.2]';
x0_2 = [-1.2, 1.0]';

% Parameters
tol = 1e-6;       % Tolerance for convergence
max_iter = 500;  % Maximum number of iterations

% Run Nelder-Mead for both starting points
[x_min1, f_min1, iter1, min_history1] = nelder_mead(rosenbrock, x0_1, tol, max_iter);
[x_min2, f_min2, iter2, min_history2] = nelder_mead(rosenbrock, x0_2, tol, max_iter);

% Display results
fprintf('\n==============================================\n');
fprintf(' TEST SU ROSENBROCK IN DUE DIMENSIONI      \n');
fprintf('==============================================\n\n');

fprintf('Starting point: [1.2, 1.2]\n');
fprintf('Minimum found: [%f, %f]\n', x_min1(1), x_min1(2));
fprintf('Function value: %f\n', f_min1);
fprintf('Iterations: %d\n\n', iter1);

fprintf('Starting point: [-1.2, 1.0]\n');
fprintf('Minimum found: [%f, %f]\n', x_min2(1), x_min2(2));
fprintf('Function value: %f\n', f_min2);
fprintf('Iterations: %d\n\n', iter2);

% Plot figures
iterations_1= 1:iter1;
iterations_2= 1:iter2;

figure;
plot(iterations_1, min_history1(1:iter1), '-o', 'DisplayName', '[1.2, 1.2]');
hold on;
plot(iterations_2, min_history2(1:iter2), '-x', 'DisplayName', '[-1.2, 1.0]');
hold off;
xlabel('Numero di Iterazioni');
ylabel('Valore della Funzione Obiettivo');
title('Convergenza del Metodo Nelder-Mead sulla Funzione di Rosenbrock');
legend show;
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTE 3 - TEST DELL'ALGORITMO SULLA FUNZIONE DI EXTENDED ROSENBROCK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%intro
matricole = [295706, 302689]; %ora 295706 é diventato 349152
rng(min(matricole));

n_NelderMead = [10, 26, 50];

extended_rosenbrock = @(x) 0.5*sum([10*(x(1:2:end).^2 - x(2:2:end)); x(1:2:end-1)-1].^2);

function xbar = initial_solution_er(n)
    
    xbar = ones(n, 1);          
    xbar(1:2:end) = -1.2;       
    
end

function X0 = generate_initial_points_er(x_bar, num_points)
    n = length(x_bar);
    X0 = repmat(x_bar, 1, num_points) + 2*rand(n, num_points) - 1;
end

function esito = is_success(f_min, tol_success)
    
    if f_min > 0 && f_min < tol_success
        esito = 1;
    elseif f_min < 0 && f_min > -tol_success
        esito = 1;
    else
        esito = 0;
    end
end

function esito = is_success_banded(f_min, tol_success, dim)
    if dim==10
        if f_min > -10.24 && f_min < -10.24 + tol_success
            esito = 1;
        elseif f_min < -10.24 && f_min > -10.24 - tol_success
            esito = 1;
        else
            esito = 0;
        end
    end

    if dim==26
        if f_min > -23.34 && f_min < -23.34 + tol_success
            esito = 1;
        elseif f_min < -23.34 && f_min > -23.34 - tol_success
            esito = 1;
        else
            esito = 0;
        end
    end

    if dim==50
        if f_min > -41.58 && f_min < -41.58 + tol_success
            esito = 1;
        elseif f_min < -41.58 && f_min > -41.58 - tol_success
            esito = 1;
        else
            esito = 0;
        end
    end

end

function q = compute_ecr(X)

    d = zeros(1,3);
    for k = 1:3
        d(k) = norm(X(:,k+1) - X(:, k), 2);
    end

    q = log(d(3) / d(2)) / log(d(2) / d(1));
end

%INIZIO TEST
max_iter = 80000;  % Maximum number of iterations
tol = 1e-6;
num_points = 10;
time_dim  = zeros(3);    % ← times collected
a = 1;

t_total = tic;

for j=n_NelderMead

    t0 = tic;

    fprintf('\n==============================================\n');
    fprintf(' TEST SU EXTENDED ROSENBROCK IN DIMENSIONE %d \n', j);
    fprintf('==============================================\n\n');

    x_bar = initial_solution_er(j);

    X0 = generate_initial_points_er(x_bar, num_points);

    % === TEST SU x_bar ===
    fprintf('\n--- TEST SU VALORE X_BAR ---\n');
    tic;
    [x_min, f_min, iter, min_hist_bar,e_rate] = nelder_mead(extended_rosenbrock, x_bar, tol, max_iter);
    t = toc;
    fprintf('f_min = %.6f\n | iter = %d | tempo = %.2fs\n', f_min, iter, t);
    rho = compute_ecr(e_rate); 
    fprintf('rho ≈ %.4f\n', rho);


    % === TEST SU 10 PUNTI CASUALI ===
    min_hist_all = cell(num_points, 1);
    successi = 0;
    for i = 1:num_points
        x0 = X0(:,i);
        fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
        tic;
        [x_min, f_min, iter, min_hist, e_rate] = nelder_mead(extended_rosenbrock, x0, tol, max_iter);
        t = toc;
        fprintf('f_min = %.6f\n | iter = %d | tempo = %.2fs\n', f_min, iter, t);
        rho = compute_ecr(e_rate);
        fprintf('rho ≈ %.4f\n', rho);
        min_hist_all{i} = min_hist;
        successi = successi + is_success(f_min, 2);
    end
 
    fprintf('\nSuccessi: %d su %d\n', successi, num_points);

    % === PLOT CONVERGENZA ===
    figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);  % finestra ampia
    hold on;
    
    % --- Plot x̄ ---
    plot(1:length(min_hist_bar), min_hist_bar, '-k', ...
        'LineWidth', 2.2, 'DisplayName', 'x̄');
    
    % --- Automatic colours ---
    colors = lines(num_points);
    
    % --- Plot 10 casual points ---
    for i = 1:num_points
        mh = min_hist_all{i};
        plot(1:length(mh), mh, '-o', ...
            'LineWidth', 1.2, ...
            'MarkerSize', 4, ...
            'Color', colors(i,:), ...
            'DisplayName', sprintf('x₀ #%d', i));
    end
    
    % --- Labels and titles ---
    xlabel('Iterazioni', 'FontSize', 13);
    ylabel('Valore funzione obiettivo', 'FontSize', 13);
    title(sprintf('Convergenza Nelder-Mead su Extended Rosenbrock (n = %d)', j), 'FontSize', 14);
    
    % --- Legenda e stile ---
    legend('show', 'Location', 'northeastoutside');
    grid on;
    set(gca, 'YScale', 'log'); 
    
    box on;
    set(gca, 'FontSize', 12);   
    hold off;

    time_dim(a) = toc(t0);
    fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             j, time_dim(a));
    a = a + 1;

end

% --------------------- TEMPO TOTALE SCRIPT ----------------------
fprintf('\n=================================================\n');
fprintf(' TABELLA TEMPISTICHE ALGORITMO EXTENDED ROSENBROCK \n');
fprintf('=================================================\n\n');

time_total = toc(t_total);

fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(1), time_dim(1));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(2), time_dim(2));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(3), time_dim(3));
fprintf('\nTempo TOTALE (tutte le dimensioni) :  %.2f  s\n', time_total);

%---  bar chart  ---
figure;
bar(categorical(string(n_NelderMead)), time_dim);
ylabel('Tempo (s)'); grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTE 3 - TEST DELL'ALGORITMO SULLA FUNZIONE Generalized Broyden tridiagonal 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

generalized_broyden = @(x) 0.5*sum(((3-2*x(2:end-1)).*x(2:end-1)+1-x(1:end-2)-x(3:end)).^2);

function xbar = initial_solution_gb(n)
    
    xbar = -ones(n+2, 1);          
    xbar(1) = 0;       
    xbar(n+2) = 0;

end

function X0 = generate_initial_points_gb(x_bar, num_points)
    n = length(x_bar);
    X0 = repmat(x_bar, 1, num_points) + 2*rand(n, num_points) - 1;
end

%INIZIO TEST
max_iter = 80000;  % Maximum number of iterations
tol = 1e-6;
num_points = 10;
time_dim  = zeros(3);    % ← times collected
a = 1;

t_total = tic;

for j=n_NelderMead

    t0 = tic;

    fprintf('\n==============================================\n');
    fprintf(' TEST SU GENERALYZED BROYDEN TRIDIAGONAL IN DIMENSIONE %d \n', j);
    fprintf('==============================================\n\n');

    x_bar = initial_solution_gb(j);

    X0 = generate_initial_points_gb(x_bar, num_points);

    % === TEST SU x_bar ===
    fprintf('\n--- TEST SU VALORE X_BAR ---\n');
    tic;
    [x_min, f_min, iter, min_hist_bar, e_rate] = nelder_mead(generalized_broyden, x_bar, tol, max_iter);
    t = toc;
    fprintf('f_min = %.6f\n | iter = %d | tempo = %.2fs\n', f_min, iter, t);
    rho = compute_ecr(e_rate);
    fprintf('rho ≈ %.4f\n', rho);

    % === TEST SU 10 PUNTI CASUALI ===
    min_hist_all = cell(num_points, 1);
    successi = 0;
    for i = 1:num_points
        x0 = X0(:,i);
        fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
        tic;
        [x_min, f_min, iter, min_hist, e_rate] = nelder_mead(generalized_broyden, x0, tol, max_iter);
        t = toc;
        fprintf('f_min = %.6f\n | iter = %d | tempo = %.2fs\n', f_min, iter, t);
        rho = compute_ecr(e_rate);
        fprintf('rho ≈ %.4f\n', rho);
        min_hist_all{i} = min_hist;
        successi = successi + is_success(f_min, 2);
    end

    fprintf('\nSuccessi: %d su %d\n', successi, num_points);

    % === PLOT CONVERGENZA ===
    figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);  % finestra ampia
    hold on;
    
    % --- Plot x̄  ---
    plot(1:length(min_hist_bar), min_hist_bar, '-k', ...
        'LineWidth', 2.2, 'DisplayName', 'x̄');
    
    % --- Colours---
    colors = lines(num_points);
    
    % --- Plot 10 casual points ---
    for i = 1:num_points
        mh = min_hist_all{i};
        plot(1:length(mh), mh, '-o', ...
            'LineWidth', 1.2, ...
            'MarkerSize', 4, ...
            'Color', colors(i,:), ...
            'DisplayName', sprintf('x₀ #%d', i));
    end
    
    % --- Labels ---
    xlabel('Iterazioni', 'FontSize', 13);
    ylabel('Valore funzione obiettivo', 'FontSize', 13);
    title(sprintf('Convergenza Nelder-Mead su Generalyzed Broyden (n = %d)', j), 'FontSize', 14);
    
    % --- Legenda e stile ---
    legend('show', 'Location', 'northeastoutside');
    grid on;
    set(gca, 'YScale', 'log');  
    
    box on;
    set(gca, 'FontSize', 12);   
    hold off;

    time_dim(a) = toc(t0);
    fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             j, time_dim(a));
    a = a + 1;

end

% --------------------- TEMPO TOTALE SCRIPT ----------------------
fprintf('\n=================================================\n');
fprintf(' TABELLA TEMPISTICHE ALGORITMO GENERALIZED BROYDEN \n');
fprintf('=================================================\n\n');

time_total = toc(t_total);

fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(1), time_dim(1));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(2), time_dim(2));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(3), time_dim(3));
fprintf('\nTempo TOTALE (tutte le dimensioni) :  %.2f  s\n', time_total);

%---  bar chart  ---
figure;
bar(categorical(string(n_NelderMead)), time_dim);
ylabel('Tempo (s)'); grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTE 3 - TEST DELL'ALGORITMO SULLA FUNZIONE BANDED TRIGONOMETRIC 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

banded_trigonometric = @(x) sum((1:length(x)-2)' .* ((1 - cos(x(2:end-1))) + sin(x(1:end-2)) - sin(x(3:end))));

function xbar = initial_solution_bt(n)
    
    xbar = ones(n+2, 1);         
    xbar(1) = 0;       
    xbar(n+2) = 0;

end

function X0 = generate_initial_points_bt(x_bar, num_points)
    n = length(x_bar);
    X0 = repmat(x_bar, 1, num_points) + 2*rand(n, num_points) - 1;
end

%INIZIO TEST
max_iter = 800000;  % Maximum number of iterations
tol = 1e-6;
num_points = 10;
time_dim  = zeros(3);    
a = 1;

t_total = tic;

for j=n_NelderMead

    t0 = tic;

    fprintf('\n==============================================\n');
    fprintf(' TEST SU BANDED TRIGONOMETRIC IN DIMENSIONE %d \n', j);
    fprintf('==============================================\n\n');

    x_bar = initial_solution_bt(j);

    X0 = generate_initial_points_bt(x_bar, num_points);

    % === TEST SU x_bar ===
    fprintf('\n--- TEST SU VALORE X_BAR ---\n');
    tic;
    [x_min, f_min, iter, min_hist_bar, e_rate] = nelder_mead(banded_trigonometric, x_bar, tol, max_iter);
    t = toc;
    fprintf('f_min = %.6f\n | iter = %d | tempo = %.2fs\n', f_min, iter, t);
    min_hist_bar = min_hist_bar(1:iter);
    rho = compute_ecr(e_rate);
    fprintf('rho ≈ %.4f\n', rho);

    % === TEST SU 10 PUNTI CASUALI ===
    min_hist_all = cell(num_points, 1);
    successi=0;
    for i = 1:num_points
        x0 = X0(:, i);
        fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
        tic;
        [x_min, f_min, iter, min_hist, e_rate] = nelder_mead(banded_trigonometric, x0, tol, max_iter);
        t = toc;
        fprintf('f_min = %.6f\n | iter = %d | tempo = %.2fs\n', f_min, iter, t);
        min_hist = min_hist(1:iter-1);
        rho = compute_ecr(e_rate);
        fprintf('rho ≈ %.4f\n', rho);
        min_hist_all{i} = min_hist;
        successi = successi + is_success_banded(f_min, 2, j);

    end

    fprintf('\nSuccessi: %d su %d\n', successi, num_points);

    % === PLOT CONVERGENZA ===
    figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);  
    hold on;
    
    % --- Plot x̄  ---
    plot(1:length(min_hist_bar), min_hist_bar, '-k', ...
        'LineWidth', 2.2, 'DisplayName', 'x̄');
    
    % --- Colours ---
    colors = lines(num_points);
    
    % --- Plot 10 casual points ---
    for i = 1:num_points
        mh = min_hist_all{i};
        plot(1:length(mh), mh, '-o', ...
            'LineWidth', 1.2, ...
            'MarkerSize', 4, ...
            'Color', colors(i,:), ...
            'DisplayName', sprintf('x₀ #%d', i));
    end
    
    % ---  labels ---
    xlabel('Iterazioni', 'FontSize', 13);
    ylabel('Valore funzione obiettivo', 'FontSize', 13);
    title(sprintf('Convergenza Nelder-Mead su Banded Trigonometric (n = %d)', j), 'FontSize', 14); 
    
    % --- Legenda e stile ---
    legend('show', 'Location', 'northeastoutside');
    grid on;
    set(gca, 'YScale', 'linear');  
    
    box on;
    set(gca, 'FontSize', 12);   
    hold off;

    time_dim(a) = toc(t0);
    fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             j, time_dim(a));
    a = a + 1;

end

% --------------------- TEMPO TOTALE SCRIPT ----------------------
fprintf('\n=================================================\n');
fprintf(' TABELLA TEMPISTICHE ALGORITMO BANDED TRIDIAGONAL \n');
fprintf('=================================================\n\n');

time_total = toc(t_total);

fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(1), time_dim(1));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(2), time_dim(2));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NelderMead(3), time_dim(3));
fprintf('\nTempo TOTALE (tutte le dimensioni) :  %.2f  s\n', time_total);

%---  bar chart  ---
figure;
bar(categorical(string(n_NelderMead)), time_dim);
ylabel('Tempo (s)'); grid on;