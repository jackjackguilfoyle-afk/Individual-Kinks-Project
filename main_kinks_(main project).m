clc; clear; close all;
% Number of gridpoints and domain size
N = 1000; L = 80;

% Spatial grid and grid spacing
x = linspace(-L,L,N)';
dx = x(2)-x(1);


%% Different initial conditions:
% % "Normal" Solution tanh (random noise and/or dilation)
% dilation = 1;
% phi0 = tanh(x./dilation);
% % without "Lorentz boost", so v0=0
% v0 = zeros(N,1);

%Moving Kink
% speed = 0.9; x0 = -10; gamma = 1/sqrt(1-speed^2);
% z = gamma*(x - x0); %kink at x0=-10 moving to the right
% phi0 = tanh(z);
% v0  = - (gamma*speed) .* sech(z).^2;

%Kink and anti-kink
speed = 0.7; x0 = -5;
gamma = 1/sqrt(1-speed^2); %speed of light = 1

z1 = gamma*(x - x0); %kink at x0=-5 moving to the right
z2 = gamma*(x + x0); %kink at x0=+5 moving to the left

phi0 =  tanh(z1) - tanh(z2) - 1;
v0   = - (gamma*speed).*sech(z1).^2 - (gamma*speed).*sech(z2).^2;

%% More initialisation
show_energy = false;   % set false for normal animation
U = @(phi) (1/2*(1-phi.^2).^2);
U_p = @(phi) (2*phi.^3 - 2*phi);

% initialize the time parameters for simulation
T = 50; dt = 0.001; time = 0:dt:T;
tspan = [0,T];

% initial state y0
y0 = [phi0; v0];

%% Laplacian matrix
% Construct the Laplacian matrix (old method)
% one_vec = ones(N,1);
% Laplacian = spdiags([one_vec, -2*one_vec, one_vec],[1,0,-1],N,N);
% % Set the Neumann boundary conditions
% % Laplacian(1,1) = -1; Laplacian(N,N) = -1;
% % Laplacian = (1/dx^2)*Laplacian;
% % still unsure about correct one lmao
% Laplacian(1,:)   = 0; Laplacian(1,1)   = -2; Laplacian(1,2)   =  2;
% Laplacian(N,:)   = 0; Laplacian(N,N)   = -2; Laplacian(N,N-1) =  2;
% Laplacian = (1/dx^2)*Laplacian;

boundary_cond = "neumann"; % define boundary condition type
Laplacian = build_laplacian(N, dx, boundary_cond);
function Lap = build_laplacian(N, dx, BC)
    one_vec = ones(N,1);
    Lap = spdiags([one_vec, -2*one_vec, one_vec], -1:1, N, N);
    switch lower(BC)
        case "neumann"
            Lap(1,:) = 0; Lap(1,1) = -2; Lap(1,2) = 2;
            Lap(end,:) = 0; Lap(end,end) = -2; Lap(end,end-1) = 2;
        case "dirichlet"
            Lap(1,:) = 0; Lap(1,1) = 1;
            Lap(end,:) = 0; Lap(end,end) = 1;
    end
    Lap = (1/dx^2)*Lap;
end

% check separately for absorbing BC
if boundary_cond == "absorbing"
    damping = build_damping(x, L, 10, 1.0);
else
    damping = zeros(N,1);
end
function damp = build_damping(x, L, width, strength)
    damp = zeros(size(x));
    left  = x < -(L - width);
    right = x >  (L - width);
    damp(left)  = strength*((x(left) + L - width)/width).^2;
    damp(right) = strength*((x(right) - (L - width))/width).^2;
end


%% Using ode113 solver
function dydt = kink_rhs(~, y, Lap, U_p, damping, BC)
    N = numel(y)/2;
    phi = y(1:N);
    v   = y(N+1:end);

    phi_t = v;

    Lap_phi = Lap * phi;
    v_t = Lap_phi - U_p(phi) - damping .* v;

    if BC == "dirichlet"
        phi_t(1) = 0; phi_t(end) = 0;
        v_t(1)   = 0; v_t(end)   = 0;
    end
    
    dydt = [phi_t; v_t];
end
[T, Y] = ode113(@(t,y) kink_rhs(t,y,Laplacian,U_p, damping, boundary_cond), tspan, y0);
state = Y(:, 1:N)';
% Plot all at once
close all;
plot(x, state(:,1),'-*');
hold on;
plot(x, state(:,round(end/4)),'-*')
plot(x, state(:,round(end/2)),'-*')
plot(x, state(:,round(3*end/4)),'-*')
plot(x, state(:, end),'-*');
xlabel('x', 'Interpreter','latex');
ylabel('$\phi$', 'Interpreter','latex');
set(gca,'fontsize',20);
title('\phi(x,t) at equidistant timestamps');
grid on;
legend('$\phi(x,t=0)$','$\phi(x,t=T/4)$','$\phi(x,t=T/2)$','$\phi(x,t=3T/4)$', '$\phi(x,t=T)$','interpreter','latex', 'Location','best')


[~, idx0] = min(abs(x));
%% Phase-space (phi, phi')
figure;
plot(Y(:, idx0), Y(:, N+idx0), 'LineWidth', 2);
xlabel('\phi(0,t)');
ylabel('\phi_t(0,t)');
set(gca,'fontsize',16);
title('Phase-space trajectory (\phi(0,t),\phi_t(0,t)) at x = 0');
grid on;


%% Energy density function
function E = energy_density(phi, v, dx, U)
    dphi = zeros(size(phi));
    dphi(2:end-1) = (phi(3:end) - phi(1:end-2)) / (2*dx);
    dphi(1) = 0; dphi(end) = 0;
    E = 0.5*v.^2 + 0.5*dphi.^2 + U(phi);
end


%% Radiation Energy
rad_energy = zeros(length(T),1);
U = @(phi) 0.5*(1 - phi.^2).^2;

for n = 1:length(T)
    phi_n = state(:,n);
    v_n   = Y(n, N+1:end)';

    E = energy_density(phi_n, v_n, dx, U);

    rad_mask = (abs(x) > 15);     % choose outer region
    rad_energy(n) = trapz(x(rad_mask), E(rad_mask));
end

figure;
plot(T, rad_energy, 'LineWidth', 2);
xlabel('time t'); ylabel('Radiation Energy');
title('Radiation energy over time');
grid on;


% Save animation as MP4 (using ode113 solver)
filename = append(boundary_cond, 'energy_kink_antikink', num2str(speed),'_animation.mp4');
% ^^ if we take "Kink and anti-kink" or "Moving Kink"
% filename = append('solution_scale=', num2str(dilation), '_kink_animation.mp4');
% ^^ if we take "Normal" Solution tanh (random noise)"
video = VideoWriter(filename, 'MPEG-4');
video.FrameRate = 10; %video.Quality = 100;
open(video);

step = max(1, floor(length(T) / 200));
figure;

for n = 1:step:length(T)
    if show_energy
        % phi plot
        subplot(2,1,2);
        plot(x, state(:,n), 'LineWidth', 2);
        ylim([-2 1.5]);
        xlim([-20 20]);
        set(gca,'fontsize',16);
        xlabel('x', 'interpreter','latex'); ylabel('$\phi$', 'Interpreter','latex');
        title(sprintf('\\phi(x,t), with initial speed v = %.3f, t = %.3f', speed, T(n)));
        legend('$\phi(x, t)$', 'Interpreter','latex');
        grid on;
        % energy plot
        subplot(2,1,1);
        phi_n = state(:,n);
        v_n = Y(n, N+1:end)';
        E = energy_density(phi_n, v_n, dx, U);
        plot(x, E, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
        xlabel('x', 'interpreter','latex'); ylabel('E', 'interpreter','latex');
        set(gca,'fontsize',16);
        ylim([-0.5 2]);
        xlim([-20 20]);
        title(sprintf('Energy density, t = %.3f', T(n)));
        legend('$E(x, t)$', 'Interpreter','latex');
        grid on;
    else
        % phi only
        plot(x, state(:,n), 'LineWidth', 2);
        hold on
        % plot(x, tanh(x), '--','LineWidth', 1);
        ylim([-2 1.5]);
        xlim([-20 20]);
        set(gca,'fontsize',16);
        xlabel('x', 'interpreter','latex'); ylabel('$\phi$', 'Interpreter','latex');
        title(sprintf('\\phi(x,t),  t = %.3f', T(n))); %for stationary kink
        %title(sprintf('\\phi(x,t) with initial speed v = %.3f,  t = %.3f', speed, T(n))); %for moving kink
        grid on;
        legend('$\phi(x, t)$', '$\tanh(x,t)$', 'Interpreter','latex'); %for stationary kink
        %legend('$\phi(x, t)$', 'Interpreter','latex'); %for moving kink
        hold off
    end
    frame = getframe(gcf);
    writeVideo(video, frame);
end
close(video);
disp(append('Saved MP4 animation as ', filename));

%colour plot
figure;
imagesc(T, x, state);
set(gca,'YDir','normal');
xlabel('Time t','Interpreter','latex');
ylabel('Space x','Interpreter','latex');
title('$\phi(x,t)$ space-time plot','Interpreter','latex');
colormap(turbo);
colorbar;
set(gca,'fontsize',20);

%waterfall plot
figure;
numSlices = 150;%fewer slices for visibility
idx = round(linspace(1, length(T), numSlices));

waterfall(x, T(idx), state(:, idx)');
xlabel('x','Interpreter','latex');
ylabel('t','Interpreter','latex');
zlabel('$\phi(x,t)$','Interpreter','latex');
title('Waterfall plot of $\phi(x,t)$','Interpreter','latex');
colormap(turbo);
set(gca,'fontsize',20);

colormap(turbo);
shading interp;
set(gca,'fontsize',20);
view([-30 30])
