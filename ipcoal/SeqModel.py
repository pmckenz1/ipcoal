#!/usr/bin/env python

from scipy.linalg import expm
import numpy as np



class SeqModel():
    def __init__(
        self,
        ttree=None,
        Q=None,
        stationary_distribution=None,
        kappa=None,
        ):

        # save the tree object if one is provided with init
        self.ttree = ttree

        # set stationary_dist, kappa, Q, mod_type
        self.set_JC()

        # save the Q matrix if one is provided with init
        if Q:
            self.Q = Q

        # record stationary distribution of base frequencies (later)
        if stationary_distribution:
            self.stationary_distribution = stationary_distribution

        # record transition to transversion ratio (later)
        self.kappa = None
        if kappa:
            self.kappa = kappa

        # base object to reference later
        self.bases = np.array([0, 1, 2, 3])  # A, C, G, T


    def set_JC(self):
        # always same stationary with JC
        self.stationary_distribution = [0.25, 0.25, 0.25, 0.25]

        # don't bother with kappa for JC
        self.kappa = None

        # always this matrix for JC
        self.Q = np.array([
            [-1., 1. / 3, 1. / 3, 1. / 3],
            [1. / 3, -1, 1 / 3, 1. / 3],
            [1. / 3, 1. / 3, -1., 1. / 3],
            [1. / 3, 1. / 3, 1. / 3, -1.],
        ])

        # save the model type
        self.mod_type = "JC"


    def set_HKY(
        self,
        kappa=1, 
        stationary_distribution=[0.25,0.25,0.25,0.25],
        ):

        # user provides stationary dist
        self.stationary_distribution = stationary_distribution
        # user provides transition / transversion ratio
        self.kappa = kappa

        # make non-normalized matrix (not in substitutions / unit of time)
        nonnormal_Q = np.array([[-(stationary_distribution[1]+stationary_distribution[2]*kappa+stationary_distribution[3]),stationary_distribution[1],stationary_distribution[2]*kappa,stationary_distribution[3]],
                                [stationary_distribution[0],-(stationary_distribution[0]+stationary_distribution[2]+kappa*stationary_distribution[3]),stationary_distribution[2],kappa*stationary_distribution[3]],
                                [stationary_distribution[0]*kappa,stationary_distribution[1],-(stationary_distribution[0]*kappa+stationary_distribution[1]+stationary_distribution[3]),stationary_distribution[3]],
                                [stationary_distribution[0],stationary_distribution[1]*kappa,stationary_distribution[2],-(stationary_distribution[0]+stationary_distribution[1]*kappa+stationary_distribution[2])]])

        # full matrix scaling factor
        mu = -1 / np.sum( np.array([nonnormal_Q[i][i] for i in range(4)]) * stationary_distribution )

        # scale by Q to get adjusted rate matrix
        self.Q = nonnormal_Q * mu

        # save the model type
        self.mod_type = "HKY"



    def run(
        self, 
        ttree=None, 
        seq_length=50,
        return_leaves=True,
        ):

        # in case we specified a tree in the init
        if not ttree:
            ttree = self.ttree

        # make a dict to hold the alignment
        alignment = {}

        # start a traversal
        for node in ttree.treenode.traverse():

            # if not the root
            if not node.is_root():
                # get branch length
                br_len=node.dist

                # get index of parent node
                parent=node.up.idx

                # get probability matrix for this branch length
                prob_mat = self._evolve_branch_probs(br_len,self.Q)

                # make all substitutions, and save the new sequence to the node index key
                alignment[node.idx] = self._substitute(alignment[parent],prob_mat)
            else:
                # if root, pull starting sequence from stationary distribution
                alignment[node.idx] = np.random.choice([0,1,2,3],p=self.stationary_distribution,size=seq_length)

        # if we just want the leaf sequences
        if return_leaves:
            # pull leaf indices from the tree
            leaves = ttree.treenode.get_leaves()

            # return the dictionary
            return({k.name: alignment[k.idx] for k in leaves})

        # or...
        else:
            # return full alignment
            return(alignment)



    def _evolve_branch_probs(self, br_len, Q):
        "exponentiate the matrix*br_len to get probability matrix"
        return(expm(Q * br_len))



    def _substitute(self, parent_seq, prob_mat):
        """
        Start with a sequence and probability matix, and make substitutions
        across the sequence.
        """
        # make an array to hold the new sequence
        new_arr = np.zeros((len(parent_seq)),dtype=np.int8)

        # for each base index...
        for i in range(len(parent_seq)):
            # store a random choice of base in each index, 
            # based on probabilities associated with starting base
            new_arr[i] = np.random.choice(self.bases, p=prob_mat[parent_seq[i]])

        # return new sequence
        return(new_arr)