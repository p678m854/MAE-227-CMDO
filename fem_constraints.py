import numpy as np

class constraints:
    n =None
    Ix=None
    Iy =None
    Iz =None
    yp =None
    zp =None
    D =None
    Area =None
    C =None
    r =None        
    theta =None
    f=None
    K0 =None



    def __init__(self,n_elements,sigma_v,u_max):
        """
        Args:
            n_elements: number of elements
               
            sigma_v: von mises stress
            u_max:maximum strain at the tip

        """
        """
        Vars:
            n:element
            Iy: Iy of beam element
            Iz: Iz of beam element
            yp: y coordinate of selected point
            zp: z coordinate of selected point
            D: diameter of the largest inscribed circle of the cross-section
            Area: cross-sectional area of element
            C: chord lengths s of element
            r: radius of curvature
            theta: angle of twist of element
            f: force vector as forces acting at each node [[fy],[fz],[mx],[my],[mz]]    
            K_inv: inverted K matrix
            K0: local stifness matrix
        """
       
        self.n_elements=n_elements
        self.n_nodes=n_elements + 1
        self.sigma_v=sigma_v
        self.u_max=u_max

    def change_point(self,yp,zp):
        constraints.yp=yp
        constraints.zp=zp
    
    def change_element(self,n,Ix,Iy,Iz,D,Area,C,r,theta,K0):
        constraints.n=n
        constraints.Ix=Ix
        constraints.Iy=Iy
        constraints.Iz=Iz
        constraints.D=D
        constraints.Area=Area
        constraints.C=C
        constraints.r=r
        constraints.theta=theta
        constraints.K0=K0

    def change_iteration(self,f,K_inv):
        constraints.f=f
        constraints.K_inv=K_inv
        

    def von_mises_lmi(self):
        n=constraints.n
        Ix=constraints.Ix
        Iy=constraints.Iy
        Iz=constraints.Iy
        yp=constraints.yp
        zp=constraints.zp
        D=constraints.D
        Area=constraints.Area
        C=constraints.C
        r=constraints.r
        theta=constraints.theta
        f=constraints.f
        K_inv=constraints.K_inv
        K0=constraints.K0
        An=np.zeros((10,self.n_nodes * 5))
        An[:,n*5:n*5+10]=np.identity(10)
        Au=np.concatenate((np.zeros((5,(self.n_nodes-1)*5)),np.identity((self.n_nodes-1)*5)))

        R_0=np.array([[np.cos(theta) ,-np.sin(theta),0 ,0             ,0            ],
                    [np.sin(theta),np.cos(theta),0 ,0             ,0            ],
                    [0             ,0            ,1 ,0             ,0            ],
                    [0             ,0            ,0 ,np.cos(theta) ,-np.sin(theta)],
                    [0             ,0            ,0 ,np.sin(theta),np.cos(theta)]],dtype='f')
        R=np.zeros([10,10],dtype='f')
        R[0:5,0:5]=R_0
        R[5:10,5:10]=R_0

        tau_coef=((1/(4*Ix))+(4/(Area*C**2)))*( D/(1+( (np.pi**2*D**4)/(16*Area**2) ) ) )*(1+0.15*( ((np.pi**2*D**4)/(16*Area**2))-(D/(2*r))  ))

        A=np.array([[0,0,0,-zp/Iy,yp/Iz,0,0,0,0,0],[0,0,tau_coef,0,0,0,0,0,0,0]],dtype='f')

        sigma=np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(A,K0),R),An),Au),K_inv),f)
        M=np.zeros((3,3))
        Qinv=np.array([[1,0],[0,1/3]])
        M[0:2,0:2]=Qinv
        M[2,0:2]=np.transpose(sigma)
        M[0:2,2]=np.transpose(sigma)
        M[2,2]=self.sigma_v**2
        return M
    
    def strain_lmi(self):
        f=constraints.f
        K_inv=constraints.K_inv
        A_=np.concatenate((np.zeros((5,(self.n_elements-1)*5)),np.identity(5)),axis=1)
        m=np.matmul(np.matmul(A_,K_inv),f)
        M=np.zeros((6,6))
        M[0:5,0:5]=np.identity(5)
        M[5,0:5]=np.transpose(m)
        M[0:5,5]=np.transpose(m)
        M[5,5]=(np.linalg.norm(self.u_max))**2
        return M

        


       
        