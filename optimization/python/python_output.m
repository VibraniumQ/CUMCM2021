load('python_res_example.mat')
load('data_A.mat')

plot3(0,0,0,'or')
xlabel('x')
ylabel('y')
zlabel('z')
hold on
plot3(pos(:,1),pos(:,2),pos(:,3),'.r')
plot3(node_pos(:,1),node_pos(:,2),node_pos(:,3),'.b')
xlim([-350 350]);ylim([-350 350]);zlim([-400 100])
