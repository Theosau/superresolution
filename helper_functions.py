def make_blocks_vectorized(x,d):
    p,m,n = x.shape
    return x.reshape(-1,m//d,d,n//d,d).transpose(1,3,0,2,4).reshape(-1,p,d,d)