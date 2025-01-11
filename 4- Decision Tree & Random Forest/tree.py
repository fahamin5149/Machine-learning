#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A good heuristic is to choose sqrt(nfeatures) to consider for each node...
import weakLearner as wl
import numpy as np
from collections import Counter
import scipy.stats as stats


#---------------Instructions------------------#

# Here you will have to reproduce the code you have already written in
# your previous assignment.

# However one major difference is that now each node non-terminal node of the
# tree  object will have  an instance of weaklearner...

# Look for the missing code sections and fill them.
#-------------------------------------------#

class Node:
    def __init__(self,klasslabel='',pdistribution=[],score=0,wlearner=None):
        """
               Input:
               --------------------------
               klasslabel: to use for leaf node
               pdistribution: posteriorprob class probability at the node
               score: split score 
               weaklearner: which weaklearner to use this node, an object of WeakLearner class or its childs...

        """

        self.lchild=None       
        self.rchild=None
        self.klasslabel=klasslabel
        self.pdistribution=pdistribution
        self.score=score
        self.wlearner=wlearner
        self.purity=0
        
    def set_childs(self,lchild,rchild):
        """
        function used to set the childs of the node
        input:
            lchild: assign it to node left child
            rchild: assign it to node right child
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        self.lchild=lchild
        self.rchild=rchild
        
        #---------End of Your Code-------------------------#

    def isleaf(self):
        """
            return true, if current node is leaf node
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if self.klasslabel!='':
            return True
        else:
            return False 
        
            
        
        #---------End of Your Code-------------------------#
    def isless_than_eq(self, X):
        """
            This function is used to decide which child node current example 
            should be directed to. i.e. returns true, if the current example should be
            sent to left child otherwise returns false.
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        # Here you will call the evaluate funciton of weaklearn on
        # the current example and return true or false...
        return self.wlearner.evaluate(X)
        
        #---------End of Your Code-------------------------#

    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split)
    

class DecisionTree:
    ''' Implements the Decision Tree For Classification With Information Gain 
        as Splitting Criterion....
    '''
    def __init__(self, exthreshold=5, maxdepth=10,
     weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):        
        ''' 
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10 
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
        ''' 
        self.maxdepth=maxdepth
        self.exthreshold=exthreshold
        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.pdist=pdist
        self.nfeattest=nfeattest
        assert (weaklearner in ["Conic", "Linear","Axis-Aligned","Axis-Aligned-Random"])
        pass
    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)            
        elif self.weaklearner== "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()    
        else:
            return wl.RandomWeakLearner(self.nsplits,self.nfeattest)

        pass
    def train(self, X, Y):
        ''' Train Decision Tree using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns:
            -----------
            Nothing
            '''
        nexamples,nfeatures=X.shape
        ## now go and train a model for each class...
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        self.root=self.build_tree(X,Y,self.maxdepth)   
        
        #---------End of Your Code-------------------------#
    
    def build_tree(self, X, Y, depth):
        """ 

            Function is used to recursively build the decision Tree 
          
            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns
            -------
            root node of the built tree...


        """
        nexamples, nfeatures=X.shape
        klasses=np.unique(Y)
#         print(f"Rows: {nexamples}, nfeatures: {nfeatures}")
#         print("Number of classes: ",len(klasses))
        current_purity = self.calculate_purity(Y)
        
        # print(f"Current purity of Y: {current_purity}")
        
        # IF NUMBER OF CLASS IN A NODE IS 1 MEANING PURE LABEL
        # IF PURITY THRESHOLD IS MET >0.95
        # IF MAX DEPT IS MET IE 10
        # THE MINIMUM NUMBER OF EXAMPLE IN A NODE IS MET
        # self, exthreshold=5, maxdepth=10,
        # weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None

        if len(klasses)==1 or depth >= self.maxdepth or nexamples <= self.exthreshold:
            # print("Pure label")
            # Ensure Y is an integer array
            Y_int = Y.astype(int)  # Convert to integer

            # Count occurrences of each label
            label_counts = np.bincount(Y_int.ravel())

            # Get the count of the most common label
            max_label_count = np.max(label_counts)

            # Calculate purity
            purity = max_label_count / len(Y)

            counts = np.bincount(Y.astype(int).flatten())
            probs = counts / np.sum(counts)
            
            most_common_label = np.argmax(label_counts) 
            # print(f"purity: {purity}, klasslabel: {final_label}")
        
            return Node(klasslabel=most_common_label, pdistribution=probs) # self,klasslabel='',pdistribution=[],score=0,wlearner=None
            
        # print("leaf node is not decided yet")

        wlearner = self.getWeakLearner()
        Node_best_score, Node_Xlidx, Node_Xridx = wlearner.train(X, Y)

        # print(f"A node is being created with feature index: {feature_index}")
        node = Node(pscore=Node_best_score, wlearner=wlearner)
        
        X_left, Y_left = X[Node_Xlidx], Y[Node_Xlidx]
        X_right, Y_right = X[Node_Xridx], Y[Node_Xridx]
        
        if len(Y_left) == 0 or len(Y_right) == 0:
            label_counts = Counter(Y)

            # Step 1: Determine the label with the maximum count
            final_label = max(label_counts, key=label_counts.get)

            # Step 2: Calculate purity as the maximum count divided by total examples
            max_count = label_counts[final_label]
            purity = max_count / len(Y)
        
            counts = np.bincount(Y.astype(int).flatten())
            probs = counts / np.sum(counts)
            
            # print(f"purity: {purity}, klasslabel: {final_label}")
        
            return Node(klasslabel=final_label, pdistribution=probs) # self,klasslabel='',pdistribution=[],score=0,wlearner=None
            
            
        node.set_childs(self.build_tree(X_left, Y_left, depth + 1),
                        self.build_tree(X_right, Y_right, depth + 1))
        
        
        return node    
        
        
    def test(self, X):
        
        ''' Test the trained classifiers on the given set of examples 
        
                   
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for each example, i.e. to which it belongs
        '''
        
        nexamples, nfeatures=X.shape
        pclasses=self.predict(X)


        return np.array(pclasses)
    def predict(self, X):
        
        """
        Test the trained classifiers on the given example X
        
                   
            Input:
            ------
            X: [1 x d] a d-dimensional test example.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z=[]
        
        for idx in range(X.shape[0]):
            
            z.append(self._predict(self.root,np.atleast_2d(X[idx])))
        
        return z 
    
    def _predict(self,node, X):
        """
            recursively traverse the tree from root to child and return the child node label
            for the given example X
        """

        if node.isleaf():
            return node.klasslabel

        # Traverse the tree according to the split condition
        if node.isless_than_eq(X):
            return self._predict(node.lchild, X)  # Go to left child
        else:
            return self._predict(node.rchild, X)  # Go to right child
    

    def __str__(self):
        """
            overloaded function used by print function for printing the current tree in a
            string format
        """
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.root)
        str += '\n---------------------------------------------------'
        return str  # self.__print(self.tree)        
    
    def calculate_purity(self, Y):
        """
        Calculates the purity of the current node based on the distribution of labels in Y.
        
        Purity here is measured by the fraction of the most common label in Y.
        """
        # Ensure Y is an integer array
        Y_int = Y.astype(int)  # Convert to integer

        # Count occurrences of each label
        label_counts = np.bincount(Y_int.ravel())

        # Get the count of the most common label
        max_label_count = np.max(label_counts)

        # Calculate purity
        purity = max_label_count / len(Y)

        return purity
     
    def _print(self, node):
        """
                Recursive function traverse each node and extract each node information
                in a string and finally returns a single string for complete tree for printing purposes
        """
        if not node:
            return
        if node.isleaf():
            return node.get_str()
        
        string = node.get_str() + self._print(node.lchild)
        return string + node.get_str() + self._print(node.rchild)
    
    def find_depth(self):
        """
            returns the depth of the tree...
        """
        return self._find_depth(self.root)
    def _find_depth(self, node):
        """
            recursively traverse the tree to the depth of the tree and return the depth...
        """
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1
    def __print(self, node, depth=0):
        """
        
        """
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)

        # Print own value
        
        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)
        
        return ret         