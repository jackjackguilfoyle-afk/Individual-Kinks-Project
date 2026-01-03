clc; clear; close all;

% grids and params
N = 500; % grid points
L = 70; % half-domain length
x = linspace(-L, L, N)'; % spatial grid (column)
dx = x(2) - x(1);

T = 120; dt = 0.01; % time final and step
tvec = 0:dt:T;
Nt = numel(tvec);

% speeds to sweep (tiny step)
speed_list = 0.21 : 1e-6 : 0.211;
Nspeed = numel(speed_list);

% storage
bounces = zeros(1, Nspeed);
phi0_ts = zeros(Nspeed, Nt); % phi at x=0 for each speed over time
capture_threshold = 15; % threshold for "capture" (for plotting)

% Laplacian
e = ones(N,1);
Lapl = spdiags([e -2*e e], -1:1, N, N);
Lapl(1,:)   = 0; Lapl(1,1) = -2; Lapl(1,2) = 2;
Lapl(end,:) = 0; Lapl(end,end) = -2; Lapl(end,end-1) = 2;
Lapl = (1/dx^2) * Lapl;

% index nearest to x=0
[~, midx] = min(abs(x));

% potential derivative U'
Uprime = @(phi) 2*phi.^3 - 2*phi;

%% main speed loop
for k = 1:Nspeed
    v_init = speed_list(k);
    gamma = 1 / sqrt(1 - v_init^2);
    a = 5;
    
    zK  = gamma * (x + a); % left kink
    zAK = gamma * (x - a); % right antikink
    phi_init = tanh(zK) - tanh(zAK) - 1;
    v0 = -(gamma*v_init).*sech(zK).^2 - (gamma*v_init).*sech(zAK).^2;
    y0 = [phi_init; v0];

    [~, Ysol] = ode113(@(t,y) kink_rhs(t,y,Lapl,Uprime), tvec, y0);
    Ysol = Ysol.';
    phi = Ysol(1:N, :);
    phi0_ts(k, :) = phi(midx, :);
    
    % bounce counting
    thr = -0.3;
    cnt = 0;
    prev = phi0_ts(k,1);
    for n = 2:Nt
        cur = phi0_ts(k,n);
        if prev >= thr && cur < thr
            cnt = cnt + 1;
        end
        prev = cur;
    end
    bounces(k) = cnt;
end

%% plotting
% bounce number vs initial speed
figure;
is_capture = bounces > capture_threshold;
plot(speed_list(~is_capture), bounces(~is_capture), 'k.', 'MarkerSize', 14); hold on;
plot(speed_list(is_capture), zeros(sum(is_capture),1), 'r.', 'MarkerSize', 14);
xlabel('Initial speed v');
ylabel('Bounce number');
title('Bounce count vs initial speed');
set(gca,'FontSize',14);
grid on;
legend('Scattering / low bounces', sprintf('Capture (> %d bounces)', capture_threshold), 'Location', 'best');

% colour map of phi(0,t)
figure;
imagesc(tvec, speed_list, phi0_ts);
axis xy;
xlabel('time t'); ylabel('initial speed v');
title('\phi(0,t) (colour map)');
colorbar;
set(gca,'FontSize',12);


%% functions
% RHS for ODEsolver
function dydt = kink_rhs(~, y, Lap, Uprime)
    Nfull = numel(y);
    Nhalf = Nfull/2;
    phi = y(1:Nhalf);
    phi_t = y(Nhalf+1:end);
    Lap_phi = Lap * phi;
    Up = Uprime(phi);
    phi_tt = Lap_phi - Up;
    
    dydt = [phi_t; phi_tt];
end
