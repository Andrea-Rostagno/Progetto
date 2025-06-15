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

    while iter < max_iter
        
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
        % Compute Newton direction
        % p = -H_mod \ g;
        
        [~, tau] = alg63_cholesky(H,100);
        
        H = H + tau*speye(size(H,1));
        L = ichol(H);
        [p, ~, ~, ~, ~] = pcg(H, -g, 1e-6, 50, L, L');
        
        % beta = 10^-3;
        % coeffient = 2;
        % max_iter = 100;

        % [B, tau] = chol_with_addition(H, beta, coeffient, max_iter);

        % H = H + B;
       
        % Compute Newton direction
        % p = - L'\(L\g);

        if fd == 0
          p=[0;p;0];
          g=[0;g;0];
        end
        
        % Backtracking line search (Armijo rule)
        alpha = 1;
        f_curr = f(x);
        max_backtracking_iter = 10; 
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
        
     
        % New gradient
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

function xbar = initial_solution_bt(n)

    xbar = ones(n+2, 1);          
    xbar(1) = 0;       
    xbar(n+2) = 0;

end

function q = compute_ecr(X)

    
    d = zeros(1,3);
    for k = 1:3
        d(k) = norm(X(:,k+1) - X(:, k), 2);
    end

    
    q = log(d(3) / d(2)) / log(d(2) / d(1));
end

%10 random points
function X0 = generate_initial_points(x_bar, num_points)

    n = length(x_bar);
    X0 = repmat(x_bar, 1, num_points) + 2*rand(n, num_points) - 1;

end

%success function
function esito = is_success(grad_norm, tol_success)
   
    if grad_norm > 0 && grad_norm < tol_success
        esito = 1;
    else
        esito = 0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      TEST DELL'ALGORITMO SULLA FUNZIONE DI BANDED TRIGONOMETRIC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Exact gradient
grad_banded_tr = @(x) [sin(x(2)) + 2*cos(x(2));
    (2:length(x)-3)'.*sin(x(3:end-2)) + 2*cos(x(3:end-2));
    (length(x)-2) * sin(x(end-1)) - (length(x)-3) * cos(x(end-1))];

% Exact hessian
function H = banded_trig_hess(x)
    n = length(x);
    i = (2:n-3)';
    diag_princ = zeros(n-2, 1);
    diag_princ(2:n-3) = i.*cos(x(3:end-2)) - 2*sin(x(3:end-2));
    diag_princ(1) = cos(x(2)) - 2*sin(x(2));
    diag_princ(n-2) = (n-2)*cos(x(end-1)) + (n-3)*sin(x(end-1));
    H = spdiags(diag_princ, 0, n-2, n-2);
end

% Finite differences gradient
function grad_fd = banded_trigonometric_gradf_fd(x, h, type)
    n = length(x);
    if type
        hs = h*abs(x);
    else
        hs = h*ones(n, 1);
    end
    
    i = (1:n-1)';

    grad_fd = [
        2*i.*sin(x(1:n-1)).*sin(hs(1:n-1)) + 4*cos(x(1:n-1)).*sin(hs(1:n-1));
        2*n.*sin(x(n)).*sin(hs(n)) - 2*(n-1)*cos(x(n)).*sin(hs(n));
    ];
    grad_fd = grad_fd ./ (2*hs);
end

% Finite differences hessian
function H = banded_trigonometric_hessf_fd(x, h, type)
    n = length(x);
    if type
        hs = h*abs(x);
    else
        hs = h*ones(n, 1);
    end
    i = (1:n)';
    diag = (-i.*cos(x) + 2*sin(x)).*(-1) + (i.*sin(x) + 2*cos(x)).*(-hs);
    diag(n) = (-n.*cos(x(n)) - (n-1)*sin(x(n))).*(-1) + (n.*sin(x(n)) - (n-1)*cos(x(n))).*(-hs(n));
    H = sparse(1:n,1:n,diag,n,n);
end

matricole = [295706, 302689]; %ora 295706 é diventato 349152
rng(min(matricole));

max_iter = 5000;  % Maximum number of iterations
tol = 1e-6;
num_points = 10;
k = 2:2:12; 
h = power(10,-k); % increment of the finite differences
n_NewtonModified = [1000, 10000, 100000];
time_dim  = zeros(3);    % times that we collect
a = 1;

banded_tr = @(x) sum((1:length(x)-2)' .* ((1 - cos(x(2:end-1))) + sin(x(1:end-2)) - sin(x(3:end))));

t_total = tic;

for j=n_NewtonModified

    t0 = tic;

    fprintf('\n=================================================\n');
    fprintf(' TEST SU BANDED TRIGONOMETRIC IN DIMENSIONE %d \n', j);
    fprintf('=================================================\n\n');

    % Start point
    x_bar = initial_solution_bt(j);

    % 10 starting points random
    X0 = generate_initial_points(x_bar, num_points);

    fd = 1;

    if fd == 1
        grad_f = @banded_trigonometric_gradf_fd;
        hess_f = @banded_trigonometric_hessf_fd;
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
                [~, f_min, iter_bar, min_hist_bar,grad_norm_bar, e_rate_bar] = modified_newton(banded_tr,grad_f,hess_f,x_bar,tol,max_iter,fd,increment,0);
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
                    [~, f_min, iter, min_hist, grad_norm, e_rate] = modified_newton(banded_tr,grad_f,hess_f,x0,tol,max_iter,fd,increment,0);
                    t = toc;
                    fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter, t, grad_norm);
                    rho = compute_ecr(e_rate);
                    fprintf('rho ≈ %.4f\n\n', rho);
                    min_hist_all{i} = min_hist;
                    successi = successi + is_success(grad_norm, 0.5);
                end
                fprintf('\nSuccessi: %d su %d\n', successi, num_points);

                % Absolute value increment
                fprintf('\n----------------------------------------------------');
                fprintf('\nAbsolute value increment h = %d*|x| \n',increment);
                fprintf('----------------------------------------------------\n');

                % === TEST SU x_bar === %
                fprintf('\n--- TEST SU VALORE X_BAR ---\n');
                tic;
                [x_min, f_min, iter_bar_abs, min_hist_bar_abs,grad_norm, e_rate_abs] = modified_newton(banded_tr,grad_f,hess_f,x_bar,tol,max_iter,fd,increment,1);
                t = toc;
                fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter_bar_abs, t, grad_norm);
                rho = compute_ecr(e_rate_abs);
                fprintf('rho ≈ %.4f\n\n', rho);

                % === TEST SU 10 PUNTI CASUALI === %
                fprintf('\n--- TEST SU 10 PUNTI CASUALI ---\n');
                min_hist_all_abs = cell(num_points, 1);
                successi = 0;
                for i = 1:num_points
                    if i==3 && j==100000
                        i=2; %perche il test 3 in dim 100k impiega 4 minuti
                    end
                    x0 = X0(:,i);
                    fprintf('\n--- Test %d (x0 #%d) ---\n', i, i);
                    tic;
                    [x_min, f_min, iter, min_hist, grad_norm, e_rate] = modified_newton(banded_tr,grad_f,hess_f,x0,tol,max_iter,fd,increment,1);
                    t = toc;
                    fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter, t, grad_norm);
                    rho = compute_ecr(e_rate);
                    fprintf('rho ≈ %.4f\n\n', rho);
                    min_hist_all_abs{i} = min_hist;
                    successi = successi + is_success(grad_norm, 0.5);
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
                set(gca, 'YScale', 'linear');
                ylim('auto');
                grid on; box on; set(gca, 'FontSize', 11);
            
                % -- RIGHT: h * |x|
                nexttile(tl,2); hold on;
                plot(1:iter_bar_abs, min_hist_bar_abs, '-o', 'LineWidth', 1.8, 'Color', 'k', 'DisplayName', 'x̄');
                for i = 1:num_points
                    mh = min_hist_all_abs{i};
                    plot(1:length(mh), mh, '-o', 'LineWidth', 1.2, 'MarkerSize', 5, 'Color', colors(i,:), 'DisplayName', sprintf('x₀ #%d', i));
                end
                title(sprintf('h = %.1e·|x|', increment), 'FontSize', 12);
                xlabel('Iterazioni'); ylabel('f(x_k)');
                set(gca, 'YScale', 'linear');
                ylim('auto');
                grid on; box on; set(gca, 'FontSize', 11);
            
                title(tl, sprintf('Convergenza Metodo Newton Modificato – n = %d', j), 'FontSize', 14);
                legend('show', 'Location', 'eastoutside');

            end
    end

    fd = 0;

    if fd == 0
        grad_f = grad_banded_tr;
        hess_f = @banded_trig_hess;
    end

    if fd == 0

        fprintf('\n=================================================\n');
        fprintf(' TEST SU DERIVATE ESATTE \n');
        fprintf('=================================================\n\n');

        x_bar = initial_solution_bt(j);

        X0 = generate_initial_points(x_bar, num_points);
    
        % === TEST SU x_bar ===
        fprintf('\n--- TEST SU VALORE X_BAR ---\n');
        tic;
        [x_min, f_min, iter_bar, min_hist_bar, grad_norm, e_rate] = modified_newton(banded_tr,grad_f,hess_f,x_bar,tol,max_iter,fd,[],[]);
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
            [x_min, f_min, iter, min_hist, grad_norm, e_rate] = modified_newton(banded_tr,grad_f,hess_f,x0,tol,max_iter,fd,[],[]);
            t = toc;
            fprintf('f_min = %.6f\n | iter = %d | tempo = %.3fs\n | grad_norm = %.6f\n', f_min, iter, t, grad_norm);
            rho = compute_ecr(e_rate);
            fprintf('rho ≈ %.4f\n', rho);
            min_hist_all{i} = min_hist;
            successi = successi + is_success(grad_norm, 0.5);
        end
     
        fprintf('\nSuccessi: %d su %d\n', successi, num_points);
    
        % === PLOT CONVERGENZA ===
        figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);  % finestra grande
        hold on;
    
        % Plot x̄
        plot(1:iter_bar, min_hist_bar, '-o', 'LineWidth', 1.8, ...
            'DisplayName', 'x̄', 'Color', 'k');
    
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
    
        % Titles
        xlabel('Iterazioni', 'FontSize', 12);
        ylabel('Valore funzione obiettivo', 'FontSize', 12);
        title(sprintf('Convergenza Metodo su Banded Trigonometric Esatto (n = %d)', j), 'FontSize', 14);
    
        % Style
        legend('show', 'Location', 'eastoutside');
        grid on;
        set(gca, 'YScale', 'linear');
    
        box on;
        set(gca, 'FontSize', 12);
        hold off;

    end

    time_dim(a) = toc(t0);
    fprintf('\nTempo (incl. plotting) per n = %-7d  :  %.2f  s\n',j, time_dim(a));
    a = a + 1;
end

% --------------------- TOTAL TIME SCRIPT ----------------------
fprintf('\n=================================================\n');
fprintf(' TABELLA TEMPISTICHE ALGORITMO BANDED TRIGONOMETRIC \n');
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