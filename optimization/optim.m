% optim
clc;clear;close all
%% Global Variables
global rotm len_edges all_edges act_up act_down len_rope refl_idx
%% Load Data
load('data_A.mat')

%% Parameters
R = 300;
eul = [-pi/180*(-36.795) pi/180*(90-78.169) 0];
rotm = eul2rotm(eul,'ZYX');
focus = [0,0,-0.534*R];

% Index Construction
node2idx = containers.Map(node_name,1:size(node_name,1));

% Find Out the Nodes that Requires Adjusting
node_pos_r = node_pos*rotm;
refl_idx = node_pos_r(:,3) < -sqrt(3)/2*R;

%% Rope
rel_rope = node_pos - act_up;
len_rope = sqrt(rel_rope(:,1).^2 + rel_rope(:,2).^2 + rel_rope(:,3).^2);

%% Graph Construction

adj = zeros(2226,2226);
for i=1:size(node_conn,1)
    temp1 = node2idx(node_conn(i,1));
    temp2 = node2idx(node_conn(i,2));
    temp3 = node2idx(node_conn(i,3));
    adj(temp1,temp2) = adj(temp1,temp2) + 1;
    adj(temp2,temp3) = adj(temp2,temp3) + 1;
    adj(temp1,temp3) = adj(temp1,temp3) + 1;
end
adj = (adj+adj')>0;
G = graph(adj);
all_edges = table2array(G.Edges);
rel_nodes = node_pos(all_edges(:,1),:) - node_pos(all_edges(:,2),:);
len_edges = sqrt(rel_nodes(:,1).^2 + rel_nodes(:,2).^2 + rel_nodes(:,3).^2);
% figure;plot(G)
%% Optimization
stretch = zeros(2226,1);

options.Algorithm = 'sqp';
options.Display = 'iter-detailed';
% options.MaxFunctionEvaluations = inf;
options.MaxIterations = 10;
options.UseParallel = false;
% options = optimoptions('fmincon','Display','iter-detailed','Algorithm','sqp');
% Boundary
lb = zeros(2226*4,1);
ub = zeros(2226*4,1);
lb(1:6678) = -350;ub(1:6678) = 350;
lb(6679:end) = -0.6;ub(6679:end) = 0.6;

x = [node_pos(:)/R;stretch];
x = fmincon(@cal_s,x,[],[],[],[],lb,ub,@mycon,options);

function [c,ceq] = mycon(x)
    R = 300;
    global len_edges all_edges len_rope act_up act_down
    pos = reshape(x(1:6678),[],3)*R;
    stretch = x(6679:end);
    
    % Length of Edges
    rel_nodes = pos(all_edges(:,1),1:3) - pos(all_edges(:,2),1:3);
    lens = rel_nodes(:,1).^2 + rel_nodes(:,2).^2 + rel_nodes(:,3).^2;
    ceq1 = lens - len_edges.^2;
    
    % Stretch Postion    
    tmp = act_up - act_down;
    vec_len = sqrt(tmp(:,1).^2 + tmp(:,2).^2 + tmp(:,3).^2);
    direction = (act_up - act_down)./[vec_len,vec_len,vec_len];
    act_up_now = act_up + direction.*[stretch,stretch,stretch];
    rel_rope = act_up_now - pos;
    ceq2 = len_rope - sqrt(rel_rope(:,1).^2 + rel_rope(:,2).^2 + rel_rope(:,3).^2);
    c = [];
    ceq = [ceq1;ceq2]; %[ceq1;ceq2];
end