import torch, pdb


class PDELoss():
    def __init__(self, rho=1, mu=1):
        super(PDELoss, self).__init__()
        self.rho = rho
        self.mu = mu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(self, outputs, inputs):
        # pdb.set_trace()

        # compute the gradients through the linear network 
        dudx, dudy = torch.autograd.grad(outputs[:,0], [inputs[0], inputs[1]], grad_outputs=torch.ones_like(outputs[:, 0]).view(-1), retain_graph=True, create_graph=True)
        dvdx, dvdy = torch.autograd.grad(outputs[:,1], [inputs[0], inputs[1]], grad_outputs=torch.ones_like(outputs[:, 1]).view(-1), retain_graph=True, create_graph=True)
        dpdx, dpdy = torch.autograd.grad(outputs[:,2], [inputs[0], inputs[1]], grad_outputs=torch.ones_like(outputs[:, 2]).view(-1), retain_graph=True, create_graph=True)

        d2udx2 = torch.autograd.grad(dudx, inputs[0], grad_outputs=torch.ones_like(dudx), retain_graph=True, create_graph=True)[0]
        d2udy2 = torch.autograd.grad(dudy, inputs[1], grad_outputs=torch.ones_like(dudy), retain_graph=True, create_graph=True)[0]

        d2vdx2 = torch.autograd.grad(dvdx, inputs[0], grad_outputs=torch.ones_like(dvdx), retain_graph=True, create_graph=True)[0]
        d2vdy2 = torch.autograd.grad(dvdy, inputs[1], grad_outputs=torch.ones_like(dvdy), retain_graph=True, create_graph=True)[0]

        continuity_error = torch.sum((dudx[:len(outputs)].squeeze() + dvdy[:len(outputs)].squeeze())**2)
        xmomentum_error = torch.sum((self.rho*(outputs[:,0]*dudx[:len(outputs)].squeeze() + outputs[:,1]*dudy[:len(outputs)].squeeze()) + dpdx[:len(outputs)].squeeze() - self.mu*(d2udx2[:len(outputs)].squeeze() + d2udy2[:len(outputs)].squeeze()))**2)
        ymomentum_error = torch.sum((self.rho*(outputs[:,0]*dvdx[:len(outputs)].squeeze() + outputs[:,1]*dvdy[:len(outputs)].squeeze()) + dpdy[:len(outputs)].squeeze() - self.mu*(d2vdx2[:len(outputs)].squeeze() + d2vdy2[:len(outputs)].squeeze()))**2)

        return continuity_error + xmomentum_error + ymomentum_error