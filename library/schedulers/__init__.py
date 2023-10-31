from torch import optim
from .bb import (
    BorzelaiBorweinScheduler,
)

SCHEDULERS = {
    'multistep': optim.lr_scheduler.MultiStepLR,
    'multisteplr': optim.lr_scheduler.MultiStepLR,
    'borzelai-borwein': BorzelaiBorweinScheduler,
    'borzelaiborweinscheduler': BorzelaiBorweinScheduler,
}
        
def select_scheduler(name, optimizer, kwargs=None):
    if kwargs is None:
        kwargs = {}
    
    if name in [None, "", "None", "none"]:
        return None
    else:
        sched_cls = SCHEDULERS[name.lower()]
        sched = sched_cls(optimizer, **kwargs)
        
        return sched