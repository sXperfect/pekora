from .grad_desc import GradientDescent
from ..losses import select_loss, select_reg

VALID_REG_P_DIST_NAME = [
    None, 
    "min_dist_var", 
    "mean_dist", 
    "NaiveAdjecentPointDistanceMinimization",
    "MinimizeAdjecentPointDistance",
]

class RegularizedGradientDescent(GradientDescent):
    def __init__(self,
        arch,
        lr:float=0.0,
        alpha=-0.25,
        loss='mse',
        loss_kwargs:dict=None,
        reg=None,
        reg_kwargs=None,
        reg_weight=1.0,
        reg_Y=None,
        reg_Y_kwargs=None,
        reg_Y_weight=1.0,
        reg_P=None,
        reg_P_kwargs=None,
        reg_P_weight=1.0,
        reg_P_dist=None,
        reg_P_dist_kwargs=None,
        reg_P_dist_weight=None,
        optimizer='adam',
        optimizer_kwargs:dict=None,
        scheduler=None,
        scheduler_kwargs:dict=None,
    ):
        super().__init__(
            arch,
            lr=lr,
            alpha=alpha,
            loss=loss,
            loss_kwargs=loss_kwargs,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

        #? Regularize
        self.reg_name = reg
        self.reg_kwargs = reg_kwargs
        self.reg_weight = reg_weight

        #? Regularize the predictions
        self.reg_Y_name = reg_Y
        self.reg_Y_kwargs = reg_Y_kwargs
        self.reg_Y_weight = reg_Y_weight
        
        #? Regularize the points matrix
        self.reg_P_name = reg_P
        self.reg_P_kwargs = reg_P_kwargs
        self.reg_P_weight = reg_P_weight
        
        #? Regularize the distance between points
        self.reg_P_dist_name = reg_P_dist
        self.reg_P_dist_kwargs = reg_P_dist_kwargs
        self.reg_P_dist_weight = reg_P_dist_weight

        self.configure_regularizer()
        
    def configure_regularizer(self):
        #? Regularize prediction Y
        assert self.reg_name in [None, "fss_spearman"], \
            f"Invalid reg_name: {self.reg_Y_name}"
                
        self.reg_f = select_loss(
            self.reg_name, 
            kwargs=self.reg_kwargs
        )

        #? Select the function that regularize the predictions
        assert self.reg_Y_name in [None, "ksa", "fss_spearman_y"], \
            f"Invalid reg_Y_name: {self.reg_Y_name}"
                
        self.reg_Y_f = select_reg(
            self.reg_Y_name, 
            kwargs=self.reg_Y_kwargs
        )
        
        #? Select the function that regularize the points matrix
        assert self.reg_P_name in [None, "nuc_norm"], \
            f"Invalid reg_P_name: {self.reg_P_name}"
        
        self.reg_P_f = select_loss(
            self.reg_P_name, 
            kwargs=self.reg_P_kwargs
        )
        
        # #? Select the function that regularize the distance between points
        # assert self.reg_P_dist_name in VALID_REG_P_DIST_NAME, \
        #     f"Invalid reg_P_dist_name: {self.reg_P_dist_name}"
            
        self.reg_P_dist_f = select_reg(
            self.reg_P_dist_name, 
            kwargs=self.reg_P_dist_kwargs
        )
        
    def training_step(self, batch, batch_idx):
        row_ids, col_ids, D_vals = self.arch.get_data(batch)

        D = self.compute_D(row_ids, col_ids)
        loss = self.loss_f(D, D_vals)
        
        self.log(
            f"main_loss",
            loss.item(),
            prog_bar=True,
        )

        if self.reg_f is not None:
            reg_loss = self.reg_f(D, D_vals)
            self.log(
                f"reg_loss",
                reg_loss.item(),
                prog_bar=True,
            )
            
            loss += self.reg_weight*reg_loss
        
        if self.reg_Y_f is not None:
            reg_Y_loss = self.reg_Y_f(D)
            self.log(
                f"reg_Y_loss",
                reg_Y_loss.item(),
                prog_bar=True,
            )
            
            loss += self.reg_Y_weight*reg_Y_loss
            
        if self.reg_P_f is not None:
            reg_P_loss = self.reg_P_f(self.arch.P)
            self.log(
                f"reg_P_loss",
                reg_P_loss.item(),
                prog_bar=True,
            )
            
            loss += self.reg_P_weight*reg_P_loss
            
        if self.reg_P_dist_f is not None:
            reg_P_dist_loss = self.reg_P_dist_f(self.arch.P)
            
            self.log(
                f"reg_P_dist_loss",
                reg_P_dist_loss.item(),
                prog_bar=True,
            )
            
            loss += reg_P_dist_loss*self.reg_P_dist_weight
        
        self.log(
            f"loss",
            loss.item(),
            prog_bar=True,
        )

        return loss