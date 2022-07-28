import torch, pdb


class PDELoss():
    def __init__(self, rho=1, mu=1):
        super(PDELoss, self).__init__()
        self.rho = rho
        self.mu = mu

    def compute_loss(self, x, y, upred, vpred, ppred):
        dudx, dudy = torch.autograd.grad(upred, [x, y], grad_outputs=torch.ones(upred.shape), retain_graph=True, create_graph=True)
        dvdx, dvdy = torch.autograd.grad(vpred, [x, y], grad_outputs=torch.ones(vpred.shape), retain_graph=True, create_graph=True)
        dpdx, dpdy = torch.autograd.grad(ppred, [x, y], grad_outputs=torch.ones(ppred.shape), retain_graph=True)

        d2udx2 = torch.autograd.grad(dudx.squeeze(), x, grad_outputs=torch.ones(dudx.squeeze().shape), retain_graph=True)[0]
        d2udy2 = torch.autograd.grad(dudy.squeeze(), y, grad_outputs=torch.ones(dudy.squeeze().shape), retain_graph=True)[0]
        d2vdx2 = torch.autograd.grad(dvdx.squeeze(), x, grad_outputs=torch.ones(dvdx.squeeze().shape), retain_graph=True)[0]
        d2vdy2 = torch.autograd.grad(dvdy.squeeze(), y, grad_outputs=torch.ones(dvdy.squeeze().shape), retain_graph=True)[0]

        continuity_error = torch.sum((dudx.squeeze() + dvdy.squeeze())**2)
        xmomentum_error = torch.sum((self.rho*(upred*dudx.squeeze() + vpred*dudy.squeeze()) + dpdx.squeeze() - self.mu*(d2udx2.squeeze() + d2udy2.squeeze()))**2)
        ymomentum_error = torch.sum((self.rho*(upred*dvdx.squeeze() + vpred*dvdy.squeeze()) + dpdy.squeeze() - self.mu*(d2vdx2.squeeze() + d2vdy2.squeeze()))**2)

        return continuity_error + xmomentum_error + ymomentum_error