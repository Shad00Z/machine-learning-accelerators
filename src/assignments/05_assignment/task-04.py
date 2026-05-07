
# ---------------------------------------------------------------------------
# Task 4 - L2-Optimized Batched Contraction
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import generate_config
    from optimizer import Optimizer
    
    # a)
    cfg = generate_config("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    print("Task a:")
    print(cfg)
    
    # b)
    #    C    M    K    N
    #    4 4096 4096 4096
    opt = Optimizer(cfg)
    
    