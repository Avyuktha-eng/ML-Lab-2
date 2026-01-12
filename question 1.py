import numpy as np
def data():
    X= np.array([[20, 6, 2], [16, 3, 6], [27, 6, 2], [19, 1, 2], [24, 4, 2], [22, 1, 5], [15, 4, 2], [18, 4, 2], [21, 1, 4], [16, 2, 4]])
    y= np.array([[386], [289], [393], [110], [280], [167], [271], [274], [148], [198]])
    return X,y

def rank_of_X(X):
    rank=np.linalg.matrix_rank(X)
    return rank

#X.c=y => c= y.X^(-1)
def pseudo_inverse(X,y):
    pseudo_inv=np.linalg.pinv(X)
    c =pseudo_inv @ y
    return pseudo_inv,c
    

def display(rank,pseudo_inv,c):
    print (f'Rank: {rank}')
    print(f'Pseudo Inverse: {pseudo_inv}')
    print(f'C: {c}')
    return

def main():
    X,y=data()
    rank=rank_of_X(X)
    pseudo_inv,c=pseudo_inverse(X,y)

    display(rank,pseudo_inv,c)

if __name__== "__main__":
        main()
