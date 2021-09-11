clc;clear;close all
load('data_A.mat')

R = 300;
eul = [-pi/180*(-36.795) pi/180*(90-78.169) 0];
rotm = eul2rotm(eul,'ZYX');
plot3(0,0,0,'or')
xlabel('x')
ylabel('y')
zlabel('z')
hold on
plot3(node_pos(:,1),node_pos(:,2),node_pos(:,3),'.b')
xlim([-350 350]);ylim([-350 350]);zlim([-400 100])

focus = [0,0,-0.534*R];

% Find Out the Nodes that Requires Adjusting
node_pos_r = node_pos*rotm;
% plot3(node_pos_r(:,1),node_pos_r(:,2),node_pos_r(:,3),'og')
refl_idx = node_pos_r(:,3) < -sqrt(3)/2*R;

refl_node = node_pos(refl_idx,:);
plot3(refl_node(:,1),refl_node(:,2),refl_node(:,3),'.r')