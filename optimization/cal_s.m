function S = cal_s(x)
    global refl_idx
    
    eul = [-pi/180*(-36.795) pi/180*(90-78.169) 0];
    rotm = eul2rotm(eul,'ZYX');
    % Focus Point
    R = 300;
    focus = [0,0,-0.534*R];
    
    pos = reshape(x(1:6678),[],3)*R;
%     plot3(pos(:,1),pos(:,2),pos(:,3),'.r')
%     drawnow
%     fprintf('update\n')
    
    pos_r = pos(refl_idx,:)*rotm;
    % Relatice Position between Nodes and Focus Point
    rel_pos = pos_r - focus;
    % Square of Distance
    dis_node_focus = rel_pos(:,1).^2 + rel_pos(:,2).^2 + rel_pos(:,3).^2;
    
    % Calculating the Loss
    loss_parab = sum(abs(sqrt(dis_node_focus) - (pos_r(:,3)+440)));
    S = loss_parab;
end