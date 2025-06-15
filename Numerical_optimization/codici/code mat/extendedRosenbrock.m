clc; clear; close all;
format long e

function [x_min, f_min, iter, min_history, grad_norm, e_rate] = modified_newton(f,grad_f,hess_f,x0,tol,max_iter,fd,h,type)

    x = x0;
    n = length(x);
    min_history = zeros(1, max_iter);
    e_rate = zeros(n,4);
    
    rho = 0.5;     % Reduction factor for backtracking
    c = 1e-4;      % Armijo condition constant
    iter = 1;

    while iter <= max_iter
        
        if fd 
            g = grad_f(x,h,type);
            H = hess_f(x,h,type);
        else  
            g = grad_f(x);
            H = hess_f(x);
        end

        % Store current function value
        min_history(iter) = f(x); 

        % Modified Hessian (ensure positive definiteness)
        %tao = max(0, sqrt(1) - min(eig(H)));
        %H_mod = H + tao * eye(n);  % Adds diagonal damping if needed

        [L, ~] = alg63_cholesky(H,100); 

        % Compute Newton direction
        %p = -H_mod \ g;

        p = - L'\(L\g);

        % Backtracking line search (Armijo rule)
        alpha = 1;
        f_curr = f(x);
        max_backtracking_iter = 40; 
        backtracking_iter = 0;

        while f(x + alpha * p) > f_curr + c * alpha * g' * p && backtracking_iter < max_backtracking_iter
            alpha = rho * alpha;
            backtracking_iter = backtracking_iter + 1;
        end
        
        % Update iterate
        x = x + alpha * p;

        f_old = f_curr;
        f_curr = f(x);  

        if iter < 5
            e_rate(:,iter) = x;
        end

        if iter >= 5
            e_rate = [e_rate(:, 2:4), x];
        end
        
        if fd 
            g = grad_f(x,h,type);
        else  
            g = grad_f(x);
        end
    
       % Check stopping criterion
        if norm(g,inf) <= tol
            break
        end

        if abs(f_curr - f_old) <= tol*max(1,abs(f_old))
            break
        end

        if f(x) <= tol
            break;
        end
        
        iter = iter + 1;
    end
    
    % Output final results
    x_min = x;
    f_min = f(x);
    min_history = min_history(1:iter);
    grad_norm = norm(g,inf);
    
end 

function [L, tau] = alg63_cholesky(A, maxIter)

    n    = size(A,1);

    % Step 1: β = ||A||_F
    beta = norm(A, 'fro');

    % % Step 2: τ0
    % if min(diag(A)) > 0
    %     tau = 0;
    % else
    %     tau = min(beta/2, 1e-1);   % non partire oltre 0.1
    % end

    % Step 2: τ iniziale
    tau0 = 1e-3;       
    tau  = 0;          

    I = speye(n);                 

    % Step 3 
    for k = 0:maxIter
        [L,flag] = chol(A + tau*I,'lower');   % L*L' = A+τI
        if flag == 0                          %  OK
            return
        end
        %tau = max(2*tau, beta/2);             
        if tau == 0
            tau = max(tau0, min(beta/2, 1e-1));   
        else
            tau = 2 * tau;                        
        end

    end

    error('Alg63: fallito dopo %d tentativi', maxIter);
end

function xbar = initial_solution_er(n)

    xbar = ones(n, 1);          
    xbar(1:2:end) = -1.2;       
    
end

function q = compute_ecr(X)

    d = zeros(1,3);
    for k = 1:3
        d(k) = norm(X(:,k+1) - X(:, k), 2);
    end

    q = log(d(3) / d(2)) / log(d(2) / d(1));
end


function X0 = generate_initial_points(x_bar, num_points)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      TEST DELL'ALGORITMO SULLA FUNZIONE DI EXTENDED ROSENBROCK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Exact gradient
function [gradf] = grad_extended_rosenbrock(x)
n = length(x);
gradf = zeros(n,1);
gradf(1:2:n-1) = 200*x(1:2:n-1).^3 - 200*x(2:2:n).*x(1:2:n-1) + x(1:2:n-1) - 1;
gradf(2:2:n) = -100*(x(1:2:n-1).^2 - x(2:2:n));
end

% Exact hessian
function [Hessf] = extended_rosenbrock_Hess(x)
n = length(x);
diags = zeros(n,3);
diags(1:2:n-1, 2) = 200*(3*x(1:2:n-1).^2 - x(2:2:n)) + 1;
diags(2:2:n, 2) = 100;
diags(1:2:n-1, 1) = -200*x(1:2:n-1);
diags(2:2:n-2, 1) = 0;
diags(2:n, 3) = diags(1:n-1, 1);
Hessf = spdiags(diags, -1:1, n, n);
end

% Finite differences gradient
function grad_fd = extended_rosenbrock_gradf_fd(x, h, type)
    n = length(x);
    if type
        hs = h*abs(x);
    else
        hs = h*ones(n, 1);
    end
    
    grad_fd = zeros(n, 1);
    grad_fd(1:2:n) = 2*x(1:2:n).*hs(1:2:n) - 2*hs(1:2:n) + 400*x(1:2:n).^3.*hs(1:2:n) + 400*x(1:2:n).*hs(1:2:n).^3 - 400*x(1:2:n).*x(2:2:n).*hs(1:2:n);
    grad_fd(2:2:n) = -200.*hs(2:2:n).*x(1:2:n).^2 + 200*hs(2:2:n).*x(2:2:n);
    grad_fd = grad_fd ./ (2*hs);
end

% Finite differences hessian
function H = extended_rosenbrock_Hessf_fd(x, h, type)
    n = length(x);
    diag = zeros(n,1); % diagonal elements
    codiag = zeros(n-1,0); % codiagonal elements
    
    % type of increment h
    if type
        hs = h*abs(x);
    else
        hs = h*ones(n, 1);
    end
    
    % construction of the tridiagonal matrix
    diag(1:2:n) = 1200*hs(1:2:n).*x(1:2:n) - 200.*x(2:2:n) + 1 + 700*hs(1:2:n).^2 + 600*x(1:2:n).^2;
    diag(2:2:n) = 100;
    codiag(1:2:n) = -100*hs(1:2:n) - 200*x(1:2:n);
    
    D = sparse(1:n,1:n,diag,n,n);
    E = sparse(2:n,1:n-1,codiag,n,n);
    
    H = E + D + E';
end

matricole = [295706, 302689]; %ora 295706 é diventato 349152
rng(min(matricole));

max_iter = 5000;  % Maximum number of iterations
tol = 1e-6;
num_points = 10;
k = 2:2:12; 
h = power(10,-k); % increment of the finite differences
n_NewtonModified = [1000, 10000, 100000];
time_dim  = zeros(3);
a = 1;

extended_rosenbrock = @(x) 0.5*sum([10*(x(1:2:end).^2 - x(2:2:end)); x(1:2:end-1)-1].^2);

t_total = tic;

for j=n_NewtonModified

    t0 = tic;

    fprintf('\n=================================================\n');
    fprintf(' TEST SU EXTENDED ROSENBROCK IN DIMENSIONE %d \n', j);
    fprintf('=================================================\n\n');

    x_bar = initial_solution_er(j);

    X0 = generate_initial_points(x_bar, num_points);

    fd = 1;

    if fd == 1
        grad_f = @extended_rosenbrock_gradf_fd;
        hess_f = @extended_rosenbrock_Hessf_fd;
    end

    if fd == 1
            fprintf('\n=================================================\n');
            fprintf(' TEST SU DERIVATE APPROSSIMATE CON DIFFERENZE FINITE \n');
            fprintf('=================================================\n\n');

            for increment = h
                
                fprintf('\n----------------------------------------------------');
                fprintf('\nDefault increment h = %d \n',increment);
                fprintf('----------------------------------------------------\n');

                % === TEST SU x_bar === %
                fprintf('\n--- TEST SU VALORE X_BAR ---\n');
                tic;
                [~, f_min, iter_bar, min_hist_bar, grad_norm_bar, e_rate_bar] = modified_newton(extended_rosenbrock,grad_f,hess_f,x_bar,tol,max_iter,fd,increment,0);
                t = toc;
                fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter_bar, t, grad_norm_bar);
                rho = compute_ecr(e_rate_bar); 
                fprintf('rho ≈ %.4f\n\n', rho);
                
                % === TEST SU 10 PUNTI CASUALI === %
                fprintf('\n--- TEST SU 10 PUNTI CASUALI ---\n');
                min_hist_all = cell(num_points, 1);
                successi = 0;
                for i = 1:num_points
                    x0 = X0(:,i);
                    fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
                    tic;
                    [~, f_min, iter, min_hist, grad_norm, e_rate] = modified_newton(extended_rosenbrock,grad_f,hess_f,x0,tol,max_iter,fd,increment,0);
                    t = toc;
                    fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter, t, grad_norm);
                    rho = compute_ecr(e_rate); 
                    fprintf('rho ≈ %.4f\n\n', rho);
                    min_hist_all{i} = min_hist;
                    successi = successi + is_success(f_min, 0.5);
                end
                fprintf('\nSuccessi: %d su %d\n', successi, num_points);

                % Absolute value increment
                fprintf('\n----------------------------------------------------');
                fprintf('\nAbsolute value increment h = %d*|x| \n',increment);
                fprintf('----------------------------------------------------\n');

                % === TEST SU x_bar === %
                fprintf('\n--- TEST SU VALORE X_BAR ---\n');
                tic;
                [x_min, f_min, iter_bar_abs, min_hist_bar_abs, grad_norm, e_rate_abs] = modified_newton(extended_rosenbrock,grad_f,hess_f,x_bar,tol,max_iter,fd,increment,1);
                t = toc;
                fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter_bar_abs, t, grad_norm);
                rho = compute_ecr(e_rate_abs); 
                fprintf('rho ≈ %.4f\n\n', rho);

                % === TEST SU 10 PUNTI CASUALI === %
                fprintf('\n--- TEST SU 10 PUNTI CASUALI ---\n');
                min_hist_all_abs = cell(num_points, 1);
                successi = 0;
                for i = 1:num_points
                    x0 = X0(:,i);
                    fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
                    tic;
                    [x_min, f_min, iter, min_hist, grad_norm, e_rate] = modified_newton(extended_rosenbrock,grad_f,hess_f,x0,tol,max_iter,fd,increment,1);
                    t = toc;
                    fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter, t, grad_norm);
                    rho = compute_ecr(e_rate); 
                    fprintf('rho ≈ %.4f\n\n', rho);
                    min_hist_all_abs{i} = min_hist;
                    successi = successi + is_success(f_min, 0.5);
                end   
                fprintf('\nSuccessi: %d su %d\n', successi, num_points);

                % === FIGURE ===
                fig = figure('Units','normalized','Position',[0.12 0.12 0.78 0.62]);
                tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
                colors = lines(num_points);
            
                % -- LEFT: h 
                nexttile(tl,1); hold on;
                plot(1:iter_bar, min_hist_bar, '-o', 'LineWidth', 1.8, 'Color', 'k', 'DisplayName', 'x̄');
                for i = 1:num_points
                    mh = min_hist_all{i};
                    plot(1:length(mh), mh, '-o', 'LineWidth', 1.2, 'MarkerSize', 5, 'Color', colors(i,:), 'DisplayName', sprintf('x₀ #%d', i));
                end
                title(sprintf('h = %.1e', increment), 'FontSize', 12);
                xlabel('Iterazioni'); ylabel('f(x_k)');
                set(gca, 'YScale', 'log'); grid on; box on; set(gca, 'FontSize', 11);
            
                % -- RIGHT: h * |x|
                nexttile(tl,2); hold on;
                plot(1:iter_bar_abs, min_hist_bar_abs, '-o', 'LineWidth', 1.8, 'Color', 'k', 'DisplayName', 'x̄');
                for i = 1:num_points
                    mh = min_hist_all_abs{i};
                    plot(1:length(mh), mh, '-o', 'LineWidth', 1.2, 'MarkerSize', 5, 'Color', colors(i,:), 'DisplayName', sprintf('x₀ #%d', i));
                end
                title(sprintf('h = %.1e·|x|', increment), 'FontSize', 12);
                xlabel('Iterazioni'); ylabel('f(x_k)');
                set(gca, 'YScale', 'log'); grid on; box on; set(gca, 'FontSize', 11);
            
                title(tl, sprintf('Convergenza Metodo Newton Modificato – n = %d', j), 'FontSize', 14);
                legend('show', 'Location', 'eastoutside');

            end
    end

    fd = 0;

    if fd == 0
        grad_f = @grad_extended_rosenbrock;
        hess_f = @extended_rosenbrock_Hess;
    end

    if fd == 0

        fprintf('\n=================================================\n');
        fprintf(' TEST SU DERIVATE ESATTE \n');
        fprintf('=================================================\n\n');

        x_bar = initial_solution_er(j);

        X0 = generate_initial_points(x_bar, num_points);
    
        % === TEST SU x_bar ===
        fprintf('\n--- TEST SU VALORE X_BAR ---\n');
        tic;
        [x_min, f_min, iter_bar, min_hist_bar, grad_norm, e_rate] = modified_newton(extended_rosenbrock,grad_f,hess_f,x_bar,tol,max_iter,fd,[],[]);
        t = toc;
        fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter_bar, t, grad_norm);
        rho = compute_ecr(e_rate); 
        fprintf('rho ≈ %.4f\n', rho);

        % === TEST SU 10 PUNTI CASUALI ===
        min_hist_all = cell(num_points, 1);
        successi = 0;
        for i = 1:num_points
            x0 = X0(:,i);
            fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
            tic;
            [x_min, f_min, iter, min_hist, grad_norm, e_rate] = modified_newton(extended_rosenbrock,grad_f,hess_f,x0,tol,max_iter,fd,[],[]);
            t = toc;
            fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter, t, grad_norm);
            rho = compute_ecr(e_rate); % se f* = 0
            fprintf('rho ≈ %.4f\n', rho);
            min_hist_all{i} = min_hist;
            successi = successi + is_success(f_min, 0.5);
        end
     
        fprintf('\nSuccessi: %d su %d\n', successi, num_points);
    
        % === PLOT CONVERGENZA ===
        figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);  
        hold on;
    
        % Plot x̄
        plot(1:iter_bar, min_hist_bar, '-o', 'LineWidth', 1.8, ...
            'DisplayName', 'x̄', 'Color', 'k');
    
        % Colori per i 10 test random
        colors = lines(num_points);
    
        % Plot dei 10 punti iniziali random
        for i = 1:num_points
            mh = min_hist_all{i};
            plot(1:length(mh), mh, '-o', ...
                'LineWidth', 1.2, ...
                'MarkerSize', 5, ...
                'Color', colors(i,:), ...
                'DisplayName', sprintf('x₀ #%d', i));
        end
    
        % Titoli e assi
        xlabel('Iterazioni', 'FontSize', 12);
        ylabel('Valore funzione obiettivo', 'FontSize', 12);
        title(sprintf('Convergenza Metodo su Extended Rosenbrock Esatto (n = %d)', j), 'FontSize', 14);
    
        % STYLE
        legend('show', 'Location', 'eastoutside');
        grid on;
        set(gca, 'YScale', 'log');
    
        box on;
        set(gca, 'FontSize', 12);
        hold off;

    end

    time_dim(a) = toc(t0);
    fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', j, time_dim(a));
    a = a + 1;
end

% --------------------- TEMPO TOTALE SCRIPT ----------------------
fprintf('\n=================================================\n');
fprintf(' TABELLA TEMPISTICHE ALGORITMO EXTENDED ROSENBROCK \n');
fprintf('=================================================\n\n');

time_total = toc(t_total);

fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NewtonModified(1), time_dim(1));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NewtonModified(2), time_dim(2));
fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n', ...
             n_NewtonModified(3), time_dim(3));
fprintf('\nTempo TOTALE (tutte le dimensioni) :  %.2f  s\n', time_total);

%---  bar chart  ---
figure;
bar(categorical(string(n_NewtonModified)), time_dim);
ylabel('Tempo (s)'); grid on;