# author: proloy das

cdef extern from "dsyevh3.h":
    int dsyevh3(double A[3][3], double Q[3][3], double w[3])

cdef extern from "dsyevq3.h":
    pass

cdef extern from "dsyevc3.h":
    pass

cdef extern from "dsytrd3.h":
    pass
