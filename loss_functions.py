import torch


class PDELoss():
    def __init__(self, rho=1, mu=1, gx=0, gy=0, gz=0):
        super(PDELoss, self).__init__()
        self.rho = rho
        self.mu = mu
        self.gx = gx
        self.gy = gy
        self.gz = gz

    def grads_velocity_component(self, velocity_component, inputs, outputs_linear):
        dudx, dudy, dudz = torch.autograd.grad(
            outputs=outputs_linear[:, velocity_component], 
            inputs=[inputs[0], inputs[1], inputs[2]], 
            grad_outputs=torch.ones_like(outputs_linear[:, velocity_component]).view(-1), 
            retain_graph=True, 
            create_graph=True
        )
        d2udx2 = torch.autograd.grad(dudx, inputs[0], grad_outputs=torch.ones_like(dudx), retain_graph=True, create_graph=True)[0]
        d2udy2 = torch.autograd.grad(dudy, inputs[1], grad_outputs=torch.ones_like(dudy), retain_graph=True, create_graph=True)[0]
        d2udz2 = torch.autograd.grad(dudy, inputs[2], grad_outputs=torch.ones_like(dudz), retain_graph=True, create_graph=True)[0]
        return dudx, dudy, dudz, d2udx2, d2udy2, d2udz2

    def compute_loss(self, inputs, outputs):

        # commpute all derivatives of velocity terms
        dudx, dudy, dudz, d2udx2, d2udy2, d2udz2 = self.grads_velocity_component(0, inputs, outputs)
        dvdx, dvdy, dvdz, d2vdx2, d2vdy2, d2vdz2 = self.grads_velocity_component(1, inputs, outputs)
        dwdx, dwdy, dwdz, d2wdx2, d2wdy2, d2wdz2 = self.grads_velocity_component(2, inputs, outputs)

        # commpute derivatives of pressure term
        dpdx, dpdy, dpdz = torch.autograd.grad(outputs[:, 3], [inputs[0], inputs[1], inputs[2]], grad_outputs=torch.ones_like(outputs[:, 1]).view(-1), retain_graph=True, create_graph=True)


        # continuity loss - NO GRAVITY FOR NOW
        u = outputs[:,0]
        v = outputs[:,1]
        w = outputs[:,2]
        continuity_residual = (dudx.squeeze() + dvdy.squeeze() + dwdz.squeeze())**2
        
        nse_x_residual = (
            self.rho * ( u*(dudx.squeeze()) + v*(dudy.squeeze()) + w*(dudz.squeeze()) ) + 
            dpdx.squeeze() + self.gx*self.rho - 
            self.mu*(d2udx2.squeeze() + d2udy2.squeeze() + d2udz2.squeeze())
        )**2

        nse_y_residual = (
            self.rho * ( u*(dvdx.squeeze()) + v*(dvdy.squeeze()) + w*(dvdz.squeeze()) ) + 
            dpdy.squeeze() + self.gy*self.rho -
            self.mu*(d2vdx2.squeeze() + d2vdy2.squeeze() + d2vdz2.squeeze())
        )**2

        nse_z_residual = (
            self.rho * ( u*(dwdx.squeeze()) + v*(dwdy.squeeze()) + w*(dwdz.squeeze()) ) + 
            dpdz.squeeze() + self.gz*self.rho -
            self.mu*(d2wdx2.squeeze() + d2wdy2.squeeze() + d2wdz2.squeeze())
        )**2


        return continuity_residual + nse_x_residual + nse_y_residual + nse_z_residual


class ReconLoss():
    def __init__(self):
        super(ReconLoss, self).__init__()

    def compute_loss(self, pts_flows_rand, outputs_linear):
        vel_interpolations_rand = pts_flows_rand.reshape((-1, 3))
        loss_recon = torch.mean((vel_interpolations_rand - outputs_linear[:, :-1])**2, dim=(-1))
        return loss_recon