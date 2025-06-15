function [x_min, f_min, iter, min_history] = modified_newton(f, grad_f, hess_f, x0, tol, max_iter,name)
% Implementation of Modified Newton Method with Armijo Backtracking Line Search

% Inputs:
%   f: Function handle of the objective function
%   grad_f: Function handle of the gradient
%   hess_f: Function handle of the Hessian
%   x0: Initial guess
%   tol: Convergence tolerance
%   max_iter: Max number of iterations

% Outputs:
%   x_min: Point that minimizes f
%   f_min: Minimum value of f
%   iter: Number of iterations performed
%   min_history: Sequence of function values (for plot)

    x = x0;
    min_history = zeros(1, max_iter);

    rho = 0.5;     % Reduction factor for backtracking
    c = 1e-4;      % Armijo condition constant
    iter = 1;

    while iter <= max_iter
        g = grad_f(x);
        H = hess_f(x);

        % Store current function value
        min_history(iter) = f(x);

        % Modified Hessian (ensure positive definiteness)
        %tao = max(0, sqrt(1) - min(eig(H)));
        %H_mod = H + tao * eye(n);  % Adds diagonal damping if needed

        [L, ~] = alg63_cholesky(H,50); 

        % Compute Newton direction
        %p = -H_mod \ g;

        p = - L'\(L\g);

        % Backtracking line search (Armijo rule)
        alpha = 1;
        f_curr = f(x);
        max_backtracking_iter = 40; 
        backtracking_iter = 0;

        if name == "bt" || name == "gb"

            p=[0;p;0];
            g=[0;g;0];

        end
       
        while f(x + alpha * p) > f_curr + c * alpha * g' * p && backtracking_iter < max_backtracking_iter
            alpha = rho * alpha;
            backtracking_iter = backtracking_iter + 1;
        end
        
        % Update iterate
        x = x + alpha * p;

        f_old = f_curr;
        f_curr = f(x);           
        g      = grad_f(x);      
    
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
        if flag == 0                          % OK
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTE 2 - TEST DELL'ALGORITMO SULLA FUNZIONE DI ROSENBROCK
% Test con due punti iniziali richiesti dall'assignment:
% [1.2, 1.2] e [-1.2, 1.0]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Test del Metodo di Newton modificato sulla funzione di Rosenbrock
clc; clear; close all;

% Funzione di Rosenbrock e sue derivate
rosenbrock = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
grad_rosen = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1));
                    200*(x(2) - x(1)^2)];

hess_rosen = @(x) [1200*x(1)^2 - 400*x(2) + 2, -400*x(1);
                   -400*x(1), 200];

% Punti iniziali
x0_1 = [1.2, 1.2];
x0_2 = [-1.2, 1.0];

% Parametri
tol = 1e-6;
max_iter = 100;

% Esecuzione algoritmo
[x_min1, f_min1, iter1, hist1] = modified_newton(rosenbrock, grad_rosen, hess_rosen, x0_1, tol, max_iter,"rn");
[x_min2, f_min2, iter2, hist2] = modified_newton(rosenbrock, grad_rosen, hess_rosen, x0_2, tol, max_iter,"rn");

% Stampa risultati
fprintf('\n==============================================\n');
fprintf(' TEST SU ROSENBROCK - METODO NEWTON MODIFICATO\n');
fprintf('==============================================\n\n');

fprintf('Starting point: [1.2, 1.2]\n');
fprintf('Minimum found: [%f, %f]\n', x_min1(1), x_min1(2));
fprintf('Function value: %f\n', f_min1);
fprintf('Iterations: %d\n\n', iter1);

fprintf('Starting point: [-1.2, 1.0]\n');
fprintf('Minimum found: [%f, %f]\n', x_min2(1), x_min2(2));
fprintf('Function value: %f\n', f_min2);
fprintf('Iterations: %d\n\n', iter2);

% Grafico
figure('Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6]);  % finestra grande
plot(1:iter1, hist1, '-o', 'DisplayName', '[1.2, 1.2]');
hold on;
plot(1:iter2, hist2, '-x', 'DisplayName', '[-1.2, 1.0]');
xlabel('Numero di Iterazioni');
ylabel('Valore della Funzione Obiettivo');
title('Convergenza Metodo Newton Modificato sulla Funzione di Rosenbrock');
legend show;
grid on;