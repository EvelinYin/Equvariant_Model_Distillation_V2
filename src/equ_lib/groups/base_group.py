import torch



class GroupBase(torch.nn.Module):

    def __init__(self, dimension, identity):
        """ Implements a group.

        @param dimension: Dimensionality of the group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group.
        """
        super().__init__()
        self.dimension = dimension
        self.register_buffer('identity', torch.Tensor(identity))

    def elements(self):
        """ Obtain a tensor containing all group elements in this group.

        """
        raise NotImplementedError()

    def product(self, h, h_prime):
        """ Defines group product on two group elements.

        @param h: Group element 1
        @param h_prime: Group element 2
        """
        raise NotImplementedError()

    def inverse(self, h):
        """ Defines inverse for group element.

        @param h: A group element from subgroup H.
        """
        raise NotImplementedError()

    def left_action_on_R2(self, h, x):
        """ Group action of an element from the subgroup H on a vector in R2.

        @param h: A group element from subgroup H.
        @param x: Vectors in R2.
        """
        raise NotImplementedError()

    def matrix_representation(self, h):
        """ Obtain a matrix representation in R^2 for an element h.

        @param h: Group element
        """
        raise NotImplementedError()

    def determinant(self, h):
        """ Calculate the determinant of the representation of a group element
        h.

        @param g:
        """
        raise NotImplementedError()

    
    def get_shared_weight_linear_weights(self, in_features, out_features):
        raise NotImplementedError()

    

    def get_channel_attention(self, q, k, v, head_dim, temperature, attn_drop=None):
        raise NotImplementedError() 


    def roll_group(self, x):
        """ Roll input feature maps defined over the group according to the group action.

        @param x: Feature maps defined over the group.
        """
        raise NotImplementedError()
    
    def trans(self, x, g):
        """ Transform weights defined over the group according to the group action.

        @param x: Weights defined over the group.
        """
        raise NotImplementedError()
    
    def roll(self, x, g):
        """ Roll weights defined over the group according to the group action.

        @param x: Weights defined over the group.
        """
        raise NotImplementedError()

    def get_canonicalization_ref(self, device, dtype):
        """ Obtain the reference elements for canonicalization.

        @param device:
        @param dtype:
        """
        raise NotImplementedError()
    
    def get_canonicalized_images(self, images, indicator):
        """ Obtain canonicalized images according to the group action.

        @param images:
        @param indicator:
        """
        raise NotImplementedError()
    
    def normalize_group_elements(self, h):
        """ Map the group elements to an interval [-1, 1]. We use this to create
        a standardized input for obtaining weights over the group.

        @param g:
        """
        raise NotImplementedError()