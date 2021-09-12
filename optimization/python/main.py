import os
import torch
import torch.nn as nn
import scipy.io as io
import logging

def set_logger(log_dir):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(log_dir, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

log_dir = './log'
set_logger(log_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_file = './for_python.mat'
data = io.loadmat(data_file)

# ## Parameters
R = 300
act_down = torch.Tensor(data['act_down'])
act_up = torch.Tensor(data['act_up'])
all_edges = torch.LongTensor(data['all_edges'].astype('int32')-1).to(device)
focus = torch.Tensor([0.0,0.0,-0.534*R]).to(device)
len_edges = torch.Tensor(data['len_edges']).squeeze().to(device)
len_rope = torch.Tensor(data['len_rope']).squeeze().to(device)
node_pos = torch.Tensor(data['node_pos'])
refl_idx = torch.nonzero(torch.LongTensor(data['refl_idx']).squeeze()-1).squeeze()
rotm = torch.Tensor(data['rotm']).to(device)

tmp = act_up - act_down
vec_len = torch.sqrt(torch.square(tmp[:,0]) + torch.square(tmp[:,1]) + torch.square(tmp[:,2]))
direction = ((act_up - act_down)/vec_len.repeat(3,1).T).to(device)
act_up = act_up.to(device)


def train_batch(targets, model, optimizer, criterion):
    # Forward pass
    outputs = model()
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

class myModel(nn.Module):
    def __init__(self, node_pos, refl_idx, rotm, all_edges, direction, focus, len_edges, act_up, len_rope):
        super(myModel, self).__init__()

        self.pos = nn.Parameter(node_pos, True)
        self.stretch = nn.Parameter(torch.zeros(2226,1),True)
        self.bias = nn.Parameter(torch.zeros(1),True)
        # self.bias = torch.Tensor([1]).to(device)
        self.refl_idx = refl_idx
        self.rotm = rotm
        self.all_edges = all_edges
        self.direction = direction
        self.focus = focus
        self.len_edges = len_edges
        self.act_up = act_up
        self.len_rope = len_rope
        self.count = 0
        self.ineq1 = nn.ReLU()
        self.ineq2 = nn.ReLU()
        self.ineq3 = nn.ReLU()

    def forward(self):
        self.count += 1
        pos_r = torch.mm(self.pos[self.refl_idx,:],self.rotm)
        # Relatice Position between Nodes and Focus Point
        rel_pos = pos_r - self.focus
        # Square of Distance
        dis_node_focus = torch.sqrt(torch.square(rel_pos[:,0]) + torch.square(rel_pos[:,1]) + torch.square(rel_pos[:,2]))
        loss = self.ineq1(torch.abs(dis_node_focus - (pos_r[:,2]+440+self.bias*2))-1)

        rel_nodes = self.pos[self.all_edges[:,0],:] - self.pos[all_edges[:,1],:]        
        lens = torch.sqrt(torch.square(rel_nodes[:,0]) + torch.square(rel_nodes[:,1]) + torch.square(rel_nodes[:,2]))
        c = self.ineq2(torch.abs(lens - self.len_edges)-0.007*self.len_edges)*100

        # Stretch Postion        
        act_up_now = self.act_up + self.direction*self.stretch.repeat(1,3)
        rel_rope = act_up_now - self.pos
        ceq = torch.abs(self.len_rope - torch.sqrt(torch.square(rel_rope[:,0]) + torch.square(rel_rope[:,1]) + torch.square(rel_rope[:,2])))*100

        stre_bound = self.ineq3(torch.abs(self.stretch) - 0.6).squeeze()

        if self.count % 1000 == 0:
            logging.info(f"loss: {torch.sum(loss):.2f} ; c: {torch.sum(c):.2f} ; ceq: {torch.sum(ceq):.2f}")
        
        # Calculating the Loss
        return torch.cat((loss,c,ceq,stre_bound),0)


if __name__ == '__main__':
    learning_rate = 0.005
    epochs = 2000000
    model = myModel(node_pos=node_pos, refl_idx=refl_idx, rotm=rotm,
                all_edges=all_edges, direction=direction, focus=focus,
                len_edges=len_edges, act_up=act_up, len_rope=len_rope).to(device)

    tgt = torch.zeros(1527+6525+2226+2226).to(device)

    # Make the loss and optimizer
    criterion = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=learning_rate)

    loss = 200000
    for epoch in range(epochs):
        if epoch % 10000 == 0:
            logging.info(f"Epoch {epoch} is started!")
        loss_ = loss
        loss = train_batch(tgt, model, optimizer, criterion)

        if epoch % 10000 == 0:
            logging.info(f"Loss after " + str(epoch).zfill(3) + f" epochs: {loss:.3f}")
            if loss<loss_:
                torch.save(model, 'best_model.pt')
                data = {'pos':model.pos.cpu().detach().numpy() , 'stretch':model.stretch.cpu().detach().numpy()}    
                io.savemat('python_res.mat',data)
