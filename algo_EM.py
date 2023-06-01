from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA


#' A product function of a matrix from an array and another matrix
#' This function allows to compute in the EM algorithm the
#' product of each matrix from an array and another matrix.
#' @param L an array.
#' @param DD a matrix.
#' @return an array composed by each computed matrices

def trans_tab(L):
    if not isinstance(L, np.ndarray):
        print("Warning: the argument is not an array.")
    
    L_new = np.zeros((L.shape[0], L.shape[0], L.shape[2]))
    
    for i in range(L.shape[2]):
        L_new[:, :, i] = np.matmul(L[:, :, i], L[:, :, i].T)
    
    return L_new


def prod_tab_mat(DD, L):
    if not isinstance(L, np.ndarray):
        print("Warning: the argument is not an array.")
    if not isinstance(DD, np.ndarray):
        print("Warning: the argument is not a matrix.")

    L_new = np.zeros((DD.shape[0], L.shape[1], L.shape[2]))

    for i in range(L.shape[2]):
        L_new[:, :, i] = np.dot(DD, L[:, :, i])

    return L_new


#' A mean function of a product between each element from an array and the transpose of each row from a matrice
#' reconstructed factor from an array and another matrix
#'
#' This function allows to compute in the EM algorithm 
#' the mean on the statistical units of the products between each element of the reconstructed factors and 
#' the corresponding instance in a bloc of observed data.
#' @param Ftilde an array.
#' @param X a matrix.
#' @return a mean value.

def moy_TildeData(Ftilde, X):
    if not isinstance(Ftilde, np.ndarray):
        print("Warning: the first argument is not an array.")
    
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        print("Warning: the second argument is not a matrix.")
    
    Ftilde_X = np.zeros((Ftilde.shape[0], X.shape[1], Ftilde.shape[2]))
    
    for i in range(Ftilde.shape[2]):
        Ftilde_X[:, :, i] = np.matmul(Ftilde[:, :, i], X[i, :].reshape(1,X.shape[1]))
    
    moy = np.mean(Ftilde_X, axis=(0, 1))
    return moy


#' A mean function of a product between each row of a matrice and the transpose of another matrice
#'
#' This function allows to compute in the EM algorithm 
#' the mean on the statistical units of the products between each 
#' row of a bloc of observed data and a row of an extra-covariate matrix.
#' @param YX a matrix.
#' @param T a matrix.
#' @return a mean value.

def moy_T_Prim_Data(YX, T):
    if not isinstance(YX, np.ndarray) or YX.ndim != 2:
        print("Warning: the first argument is not a matrix.")
    
    if not isinstance(T, np.ndarray) or T.ndim != 2:
        print("Warning: the second argument is not a matrix.")
    
    YX_Tprim = np.zeros((YX.shape[1], T.shape[1], YX.shape[0]))
    
    for i in range(YX.shape[0]):
        YX_Tprim[:, :, i] = np.dot(YX[i, :].T.reshape(YX.shape[1],1), T[i, :].reshape(1,T.shape[1]))
    
    moy = np.mean(YX_Tprim, axis=(0, 1))
    return moy

#' A mean function of a product between each row of a matrice and its transpose
#' This function allows to compute in the EM algorithm 
#' the mean on the statistical units of the products between each 
#' row of an extra-covariate matrix and a row of the same extra-covariate matrix.
#' @param T a matrix.
#' @param Tbis a matrix.
#' @return a mean value.

def moy_T_T_Prim(T, Tbis):
    if not isinstance(T, np.ndarray) or T.ndim != 2:
        print("Warning: the first argument is not a matrix.")
    
    if not isinstance(Tbis, np.ndarray) or Tbis.ndim != 2:
        print("Warning: the second argument is not a matrix.")
    
    T_T_prim = np.zeros((T.shape[1], Tbis.shape[1], T.shape[0]))
    
    for i in range(T.shape[0]):
        T_T_prim[:, :, i] = np.matmul(T[i, :].T, Tbis[i, :])
    
    moy = np.mean(T_T_prim, axis=(0, 1))
    return moy


#' A mean function of a product between each element from an array and the transpose from another array
#' This function allows to compute in the EM algorithm 
#' the mean on the statistical units of the products between each 
#' element of an explanory reconstructed factor and each element from 
#' a dependent reconstructed factor.
#' @param Ftilde an array corresponding to the explanatory factors.
#' @param Gtilde an array corresponding to the dependent factor.
#' @return a mean value.

def moy_fg_tilde(Ftilde, Gtilde):
    if not isinstance(Ftilde, np.ndarray) or Ftilde.ndim != 3:
        print("Warning: the first argument is not an array.")
    
    if not isinstance(Gtilde, np.ndarray) or Gtilde.ndim != 3:
        print("Warning: the second argument is not an array.")
    
    FGtilde = np.zeros((Ftilde.shape[0], Gtilde.shape[1], Ftilde.shape[2]))
    
    for i in range(Ftilde.shape[2]):
        FGtilde[:, :, i] = np.matmul(Ftilde[:, :, i], Gtilde[:, :, i].T)
    
    moy = np.mean(FGtilde, axis=(0, 1))
    return moy

#' A product function of each element from an array and each row of a matrix
#' This function allows to compute in the EM algorithm 
#' the products between each matrix from an array and 
#' each row of a matrix.
#' @param mat a matrix.
#' @param L an array.
#' @return an array.

def prod_tab_rowmat(mat, L):
    if not isinstance(L, np.ndarray) or L.ndim != 3:
        print("Warning: the argument is not an array.")
    
    if L.shape[2] != mat.shape[0]:
        print("Warning: the number of matrices from L is different from the number of rows.")
    
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        print("Warning: the argument is not a matrix.")
    
    L_new = np.zeros_like(mat)
    
    for i in range(mat.shape[0]):
        L_new[i, :] = mat[i, :] * L[:, :, i]
    
    return L_new

#' A scale and center function adapted to vectors
#' This function allows to scale and center a vector.
#' @param v_test a vector.
#' @return a vector.

def cr(v_test):
    if not isinstance(v_test, np.ndarray) or v_test.ndim != 1 or v_test.dtype != np.float64:
        print("Warning: the argument is not a vector and you need a numeric class.")
    
    esp = np.mean(v_test)
    ectyp = np.sqrt(np.var(v_test) * (len(v_test) - 1) / len(v_test))
    v_test_centre_redui = (v_test - np.repeat(esp, len(v_test))) / np.repeat(ectyp, len(v_test))
    
    return v_test_centre_redui


 #### Function arguments
    # X1      : (matrix) block of data concerning the first factor 
    # X2      : (matrix) block of data concerning the second factor 
    # Y       : (matrix/vecteur) response variable
    # epsilon : (real) criterion of convergence 
    # nb_it   : (integer) number of iteration
    #nb_it = 100  # limite du nombre d'itirations


def emsem_function(Y,X1,X2,T,T1,T2,epsilon=10**(-3),nb_it=100):
   
    #----ENTRE DE PARAMETRES----
   
    qX1 = np.shape(X1)[1]
    qX2 = np.shape(X2)[1]
    qY = np.shape(Y)[1]
    qT= np.shape(T)[1] #New
    qT1= np.shape(T1)[1] #New
    qT2= np.shape(T2)[1] #New
    kF1 = 1 # CHOIX SELON NOTRE REDACTION : SI != 1 LES RESULTATS NE SONT PAS GARANTIS!
    kF2 = 1
    kG = 1 # CHOIX SELON NOTRE REDACTION : SI != 1 LES RESULTATS NE SONT PAS GARANTIS!
    n = np.shape(Y)[0]
    nb_fact = kF1+kF2+kG

    #----PARAMETRES A ESTIMER : CREATION OBJETS----
  
    A1 = np.zeros((kF1, qX1))
    A2 = np.zeros((kF2, qX2))
    B = np.zeros((kG, qY))
    C1 = 0
    C2 = 0
    #New
    D = np.zeros((qT, qY))
    D1 = np.zeros((qT1, qY))
    D2 = np.zeros((qT2, qY))
    Psi_X1 = np.zeros((qX1, qX1))
    Psi_X2 = np.zeros((qX2, qX2))
    Psi_Y = np.zeros((qY, qY))
    sigma2_X1_chap = 0
    sigma2_X2_chap = 0
    sigma2_Y_chap = 0

    #Objets necessaires a l'estimation des sigma2 chapeau 

    dud1_X1 = np.zeros((n, qX1))
    res1_X1 = np.zeros((n, 1))
    res2_X1 = np.zeros((n, 1))
    res3_X1 = np.zeros((n, 1))
    res_X1_chap = np.zeros((n, 1))
    
    dud1_X2 = np.zeros((n, qX2))
    res1_X2 = np.zeros((n, 1))
    res2_X2 = np.zeros((n, 1))
    res3_X2 = np.zeros((n, 1))
    res_X2_chap = np.zeros((n, 1))
    
   
    dud1_Y = np.zeros((n, qY))
    res1_Y = np.zeros((n, 1))
    res2_Y = np.zeros((n, 1))
    res3_Y = np.zeros((n, 1))
    res_Y_chap = np.zeros((n, 1))


    A1_it = np.zeros((kF1, qX1))
    A2_it = np.zeros((kF2, qX2))
    B_it = np.zeros((kG, qY))
    C1_it = 0
    C2_it = 0


    D_it = np.zeros((qT, qY))
    D1_it = np.zeros((qT1, qY))
    D2_it = np.zeros((qT2, qY))
    sigma2_X1_chap_it = 0
    sigma2_X2_chap_it = 0
    sigma2_Y_chap_it = 0



    #-----INITIALISATION PAR ACP ET REGRESSION----- 
    
    #Regression pour estimation de D1_sim


    regX1_Fact = LinearRegression(fit_intercept=False).fit(T1, X1)
    ini_D1 = regX1_Fact.coef_[:qT1,] #New
    #Regression pour estimation de A1_sim et sigma2X1

    # Subtract the first-stage predicted values from X1
    X1_res = X1 - np.dot(T1, ini_D1)

    # Perform PCA on the residuals and extract the first principal component
    pcaX1_Fact = PCA(n_components=1, svd_solver='full')
    cl1X1_Fact = pcaX1_Fact.fit_transform(X1_res)
    cl1X1_Fact = cl1X1_Fact.squeeze() # Remove the redundant dimension

    # Standardize the PCA scores
    cl1X1_Fact = scale(cl1X1_Fact)

    # Regress X1 on the first principal component
    regX1_Fact = LinearRegression(fit_intercept=False).fit(cl1X1_Fact.reshape(-1,1), X1_res)

    # Fit the first-stage regression of X1 on T1

    regX1_Fact = LinearRegression(fit_intercept=False).fit(cl1X1_Fact.reshape(-1,1), X1_res)

    # Extract the coefficients and variance estimate from the regression
    ini_A1_Fact = regX1_Fact.coef_.reshape(1,-1)
    sig_lm_qX1_Fact = regX1_Fact.predict(T1) - X1
    ini_sigma2X1_Fact = np.mean(sig_lm_qX1_Fact ** 2)

    #Pour eq de X2
    #Regression pour estimation de D2_sim

    # Fit the first-stage regression of X2 on T2
    regX2_Fact = LinearRegression(fit_intercept=False).fit(T2, X2)

    # Extract the coefficients from the regression
    ini_D2 = regX2_Fact.coef_.reshape(-1,1)[:qT2]  #New

    #Regression pour estimation de A2_sim et sigma2X2
    # Subtract the effect of T2 from X2
    X2_res = X2 - T2 @ ini_D2.T

    # Calculate the first principal component of X2
    pca_cl1X2_Fact = PCA(n_components=1)
    cl1X2_Fact = pca_cl1X2_Fact.fit_transform(X2_res)
    cl1X2_Fact = cl1X2_Fact.ravel()

    # Fit the regression of X2 on the first principal component of X2
    regX2_Fact = LinearRegression(fit_intercept=False).fit(cl1X2_Fact.reshape(-1,1), X2_res)

    # Extract the coefficients and variance estimate from the regression
    ini_A2_Fact = regX2_Fact.coef_.reshape(1,-1)
    sig_lm_qX2_Fact = regX2_Fact.predict(T2) - X2
    ini_sigma2X2_Fact = np.mean(sig_lm_qX2_Fact ** 2)


    #Regression pour estimation de D_sim

    # Fit a linear regression of Y on T
    regY_Fact = LinearRegression(fit_intercept=False).fit(T, Y)

    # Get the coefficients of T
    ini_D = regY_Fact.coef_[:qT]  #New 

    #Regression pour estimation de B_sim et sigma2Y
    Y_res = Y - T @ ini_D.T
    pca_cl1Y_Fact = PCA(n_components=1)
    cl1Y_Fact = pca_cl1Y_Fact.fit_transform(Y_res)
    cl1Y_Fact = cl1Y_Fact.ravel()
    
    # Fit a linear regression model of cl1Y_Fact on cl1X1_Fact and cl1X2_Fact
    reg_cl1Y = LinearRegression(fit_intercept=False).fit(np.column_stack((cl1X1_Fact, cl1X2_Fact)), cl1Y_Fact)

    # Divide cl1Y_Fact by the standard error of the regression residuals
    cl1Y_Fact /= np.mean((reg_cl1Y.predict(np.column_stack((cl1X1_Fact, cl1X2_Fact))) - cl1Y_Fact) ** 2)   

    # Régression 
    regY_Fact = sm.OLS(Y - np.dot(T, ini_D.T), cl1Y_Fact - 1 ).fit() 
    # Coefficients de régression 
    ini_B_Fact = np.matrix(regY_Fact.params).reshape(kG, qY) 
    # Erreurs standard de régression 
    sig_lm_qY_Fact = np.repeat(np.nan, qY) 

    for i in range(qY): 
        sig_lm_qY_Fact[i] = regY_Fact.bse[i] 

    # Initialisation de sigma2Y 
    ini_sigma2Y_Fact = np.mean(np.array(sig_lm_qY_Fact**2),axis=0) 


    #Regression de l'initialisation de F sur celle de G pour estimation C_sim 
    regG_Fact = LinearRegression(fit_intercept=False).fit(np.column_stack((cl1X1_Fact, cl1X2_Fact)), cl1Y_Fact)
    ini_C1_Fact = regG_Fact.coef_[0] #initialisation du parametre C1
    ini_C2_Fact = regG_Fact.coef_[1] #initialisation du parametre C2

        # Debut initialisation


    A1 = ini_A1_Fact
    A2 = ini_A2_Fact
    B = ini_B_Fact
    C1 = ini_C1_Fact
    C2 = ini_C2_Fact
    D = ini_D  # New
    D1 = ini_D1  # New
    D2 = ini_D2  # New
    sigma2_X1_chap = ini_sigma2X1_Fact
    sigma2_X2_chap = ini_sigma2X2_Fact
    sigma2_Y_chap = ini_sigma2Y_Fact

    
    Psi_X1 = np.diag(np.full(qX1,sigma2_X1_chap))  # necessary for loop to calculate M-i and Sigma_i of the distribution that gives F_tilde, G_tilde, Phi_tilde, and Gamma_tilde
    Psi_X2 = np.diag(np.full(qX2,sigma2_X2_chap))
    Psi_Y = np.diag(np.full(qY, sigma2_Y_chap))

   
      # FIN INITIALISATION            

    it = 0  # ATTENTION: `it = 0` creates objects of size zero and may cause bugs in `while{}` from line 118 if `it` doesn't increment to `it = 0 + 1`

    # We create a vector that will store the differences from one iteration to another to observe the convergence evolution
    diffgraph = np.zeros((nb_it - 1, 1))

    diff = 1  # Initial value of the parameter measuring the change in the value of theta^[t] from one iteration to another

    # Therefore, the initialization corresponds to `diffgraph` as follows:
    diffgraph[0] = 1

    detE2 = np.zeros((nb_it, 1))

    while diff > epsilon and it + 1 < nb_it:
        
        it = it + 1

        # Calculate phi_tilde, gamma_tilde, f_tilde, and g_tilde for each individual i:
        # Need to first define M_i and GAMMA_i, the mean vector and covariance matrix of the normal distribution of H_i|Z_i
        
        ##              Definitions of elements needed for parameter definition at iteration [t+1]:                   ##
        
        # When the parameters are initialized as matrices and the theoretical dimensions are preserved,
        # use this code for E1 and E2, or codes similar to the theory without the t() used when initializing them as numeric to recover the theoretical dimensions.

        E1_1 = np.column_stack((((C1) ** 2 + (C2) ** 2 + 1) * B,
                              C1 * A1,
                              C2 * A2))  
        
        E1_2 = np.column_stack((C1 * B,
                              A1,
                              np.zeros((kG,qX2))))
        
        E1_3 = np.column_stack((C2 * B,
                              np.zeros((kG,qX1)),
                              A2))
        
        E1 = np.vstack((E1_1, E1_2, E1_3))
        
        
        E2_1 = np.column_stack((((C1) ** 2 + (C2) ** 2 + 1) * B.T @ B + Psi_Y,
                              C1 * B.T @ A1,
                              C2 * B.T @ A2)) 
        E2_2 = np.column_stack((C1 * A1.T @ B,
                              A1.T @ A1 + Psi_X1,
                              np.zeros((qX1, qX2))))
        E2_3 = np.column_stack((C2 * A2.T @ B,
                              np.zeros((qX1, qX2)),
                              A2.T @ A2 + Psi_X2))
        
        E2 = np.vstack((E2_1, E2_2, E2_3))  # superpose the rows E_1, E_2, and E_3 to construct the matrix E, i.e., D2
        
        detE2[it, :] = np.linalg.det(E2)
        
        E3 = np.zeros((qY + qX1 + qX2, 1, n))
        # New
        for i in range(n):
            E3[:, :, i] = np.concatenate(((Y[i, :] -T[i, :] @ D.T ).T,
                                          (X1[i, :] - T1[i, :] @ D1.T ).T,
                                          (X2[i, :] -  T2[i, :] @ D1.T).T), axis=0).reshape(qY + qX1 + qX2, 1)



        # Codons GaMMA_i

        E4 = np.array([[(C1) ** 2 + (C2) ** 2 + 1, C1, C2],[C1, 1, 0],[C2, 0, 1]])

        GAMMA = E4 - np.matmul(np.matmul(E1, np.linalg.inv(E2)), E1.T)

        EE = np.matmul(E1, np.linalg.inv(E2))
        tEE = np.matmul(np.linalg.inv(E2).T, E1.T)
        
        fg =  prod_tab_mat(EE,E3) 

        # fg is an array from i=1 to n with elements fg[,,i]
        # each fg[,,i] contains the first kG rows m_{1i}, which represent the mean of h_i|z_i
        # the next kF1 rows contain m_{2i}, and the last kF2 rows contain m_{3i}


        # G_tilde
        G_tilde = fg[0:kG,:, :]
        # G_tilde_bar
        G_tilde_bar = np.mean(G_tilde, axis=(0, 1))

        # F1_tilde
        F1_tilde = fg[kG:kG + kF1, :, :]
        # F1_tilde_bar
        F1_tilde_bar = np.mean(F1_tilde, axis=(0, 1))

        # F2_tilde
        F2_tilde = fg[kG + kF1:kG + kF1 + kF2, :, :]
        # F2_tilde_bar
        F2_tilde_bar = np.mean(F2_tilde, axis=(0, 1))

        # Calculate G_tilde_Y_bar
        G_tilde_Y_bar = moy_TildeData(G_tilde, Y).T

        # Calculate G_tilde_T_prim_bar
        G_tilde_T_prim_bar = moy_TildeData(G_tilde , T)

        # Calculate G_tilde_T_bar
        G_tilde_T_bar = moy_TildeData(G_tilde,T).T

        # Calculate F1_tilde_X1_bar
        F1_tilde_X1_bar = moy_TildeData(F1_tilde , X1).T

        # Calculate F2_tilde_X2_bar
        F2_tilde_X2_bar = moy_TildeData(F2_tilde , X2).T

        # Calculate F1_tilde_T1_prim_bar
        F1_tilde_T1_prim_bar = moy_TildeData(F1_tilde , T1)

        # Calculate F2_tilde_T2_prim_bar
        F2_tilde_T2_prim_bar = moy_TildeData(F2_tilde ,T2)

        # Calculate F1_tilde_T1_bar
        F1_tilde_T1_bar = moy_TildeData(F1_tilde,T1).T

        # Calculate F2_tilde_T2_bar
        F2_tilde_T2_bar = moy_TildeData(F2_tilde , T2).T

        # Calculate Y_T_prim_bar
        Y_T_prim_bar = moy_T_Prim_Data(Y , T)

        # Calculate X1_T1_prim_bar
        X1_T1_prim_bar = moy_T_Prim_Data(X1 , T1)

        # Calculate X2_T2_prim_bar
        X2_T2_prim_bar = moy_T_Prim_Data(X2,T2)

        # Calculate moy_G_tilde
        moy_G_tilde = np.mean(G_tilde, axis=(0, 1))

        # Calculate moy_F1_tilde
        moy_F1_tilde = np.mean(F1_tilde, axis=(0, 1))

        # Calculate moy_F2_tilde
        moy_F2_tilde = np.mean(F2_tilde, axis=(0, 1))

        # Calculate G_tilde_bar_2
        G_tilde_bar_2 = moy_G_tilde @ (moy_G_tilde).T

        # Calculate F1_tilde_bar_2
        F1_tilde_bar_2 = moy_F1_tilde @ (moy_F1_tilde).T

        # Calculate F2_tilde_bar_2
        F2_tilde_bar_2 = moy_F2_tilde @ (moy_F2_tilde).T

        # Calculate F1G_tilde_bar
        F1G_tilde_bar = moy_fg_tilde(F1_tilde , G_tilde)

        # Calculate F2G_tilde_bar
        F2G_tilde_bar = moy_fg_tilde(F2_tilde , G_tilde)

        # Calculate F1F2_tilde_bar
        F1F2_tilde_bar = moy_fg_tilde(F1_tilde , F2_tilde)

        # GAMMA_12_bar
        GAMMA_12_bar = GAMMA[:kG, kG:(kG+kF1)]

        # GAMMA_13_bar
        GAMMA_13_bar = GAMMA[:kG, (kG+kF1):(kG+kF1+kF2)]

        # GAMMA_23_bar
        GAMMA_23_bar = GAMMA[kG:(kG+kF1), (kG+kF1):(kG+kF1+kF2)]


        # Initialize Gamma_tilde
        Gamma_tilde = np.zeros((kG, kG, n))

        # Calculate Gamma_tilde
        for i in range(n):
            E3_squared = np.power((EE @ E3[:, :, i]),2)
            Gamma_tilde[:, :, i] = E3_squared[:kG, :] + GAMMA[:kG, :kG]

        # Initialize Phi1_tilde
        Phi1_tilde = np.zeros((kF1, kF1, n))

        # Calculate Phi1_tilde
        for i in range(n):
            E3_squared = np.power((EE @ E3[:, :, i]),2)
            Phi1_tilde[:, :, i] = E3_squared[kG:kG+kF1, :] + GAMMA[kG:kG+kF1, kG:kG+kF1]

        # Initialize Phi2_tilde
        Phi2_tilde = np.zeros((kF2, kF2, n))

        # Calculate Phi2_tilde
        for i in range(n):
            E3_squared = np.power((EE @ E3[:, :, i]),2)
            Phi2_tilde[:, :, i] = E3_squared[kG+kF1:kG+kF1+kF2, :] + GAMMA[kG+kF1:kG+kF1+kF2, kG+kF1:kG+kF1+kF2]



        # Calculate Gamma_tilde_bar
        Gamma_tilde_bar = np.mean(Gamma_tilde, axis=(0, 1))

        # Calculate Phi1_tilde_bar
        Phi1_tilde_bar = np.mean(Phi1_tilde, axis=(0, 1))

        # Calculate Phi2_tilde_bar
        Phi2_tilde_bar = np.mean(Phi2_tilde, axis=(0, 1))

        # Calculate Phi1Phi2_tilde_bar
        Phi1Phi2_tilde_bar = moy_fg_tilde(Phi1_tilde, Phi2_tilde)

        # Calculate T_T_prim_bar
        T_T_prim_bar = moy_T_T_Prim(T, T)

        # Calculate T1_T1_prim_bar
        T1_T1_prim_bar = moy_T_T_Prim(T1, T1)

        # Calculate T2_T2_prim_bar
        T2_T2_prim_bar = moy_T_T_Prim(T2, T2)

        #Stock des valeurs des parametres a l'iteration it avant actualisation et l'obtention de leur valeur pour it+1
        #Objectif = nous permettre de calculer la difference de valeur d'une iteration a l'autre pour le critere d'arret
        #THETA[it] :
        
        A1_it = A1
        A2_it = A2
        B_it = B
        C1_it = C1
        C2_it = C2
        D_it = D #New
        D1_it = D1 #New
        D2_it = D2 #New
        sigma2_X1_chap_it =  sigma2_X1_chap
        sigma2_X2_chap_it =  sigma2_X2_chap
        sigma2_Y_chap_it = sigma2_Y_chap
        
        ##############################################################
        #THETA[it+1,] 
        #Parametres a l'iteration it+1 :
        # Calcul de B
        B = np.transpose((G_tilde_Y_bar - Y_T_prim_bar @ np.linalg.pinv(T_T_prim_bar) @ G_tilde_T_bar) @ 
                 np.linalg.pinv(Gamma_tilde_bar - G_tilde_T_prim_bar @ np.linalg.pinv(T_T_prim_bar) @ G_tilde_T_bar))


        A1 = np.transpose((F1_tilde_X1_bar - X1_T1_prim_bar@np.linalg.inv(T1_T1_prim_bar) @ F1_tilde_T1_bar ) @
                np.linalg.solve(Phi1_tilde_bar - F1_tilde_T1_prim_bar, np.linalg.inv(T1_T1_prim_bar) @ F1_tilde_T1_bar))

        A2 = np.transpose((F2_tilde_X2_bar - X2_T2_prim_bar @ np.linalg.inv(T2_T2_prim_bar) @ F2_tilde_T2_bar ) @
                np.linalg.solve(Phi2_tilde_bar - F2_tilde_T2_prim_bar, np.linalg.inv(T2_T2_prim_bar) @ F2_tilde_T2_bar))
        
        C1 = np.linalg.solve( Phi1Phi2_tilde_bar - (GAMMA_23_bar + F1F2_tilde_bar)^2, ((GAMMA_12_bar + F1G_tilde_bar)*Phi2_tilde_bar - (GAMMA_13_bar + F2G_tilde_bar)*(GAMMA_23_bar + F1F2_tilde_bar)))
        
        C2 = np.linalg.solve( Phi1Phi2_tilde_bar - (GAMMA_23_bar + F1F2_tilde_bar)^2, ((GAMMA_13_bar + F2G_tilde_bar)*Phi1_tilde_bar - (GAMMA_12_bar + F1G_tilde_bar)*(GAMMA_23_bar + F1F2_tilde_bar)))
        
        D = np.transpose((Y_T_prim_bar - np.transpose(B) @ G_tilde_T_prim_bar) @ np.linalg.inv(T_T_prim_bar)) 

        D1 = np.transpose((X1_T1_prim_bar - np.transpose(A1) @ F1_tilde_T1_prim_bar) @ np.linalg.solve(T1_T1_prim_bar))

        D2 = np.transpose((X2_T2_prim_bar - np.transpose(A2) @ F2_tilde_T2_prim_bar) @ np.linalg.solve(T2_T2_prim_bar))

        ####################################################
        #    sigma2_Y_chap
        #######################################################
        for i in range(n) : 
          dud1_Y[i,] = Y[i,]-np.transpose(D) @ T[i,]

        res1_Y = np.apply_along_axis(np.sum, axis=1, arr=dud1_Y**2)
        res2_Y = np.array([np.sum(B**2)*Gamma_tilde])
        tab_res3_Y = np.zeros(shape=(n, 1, n))

        for i in range(n):
          tab_res3_Y[:,:,i] = dud1_Y[i,] @ np.transpose(B) @ G_tilde[:,:,i]

        res3_Y = np.apply_over_axes(np.mean, tab_res3_Y, [1, 2]).reshape(n)
        res_Y_chap = res1_Y + res2_Y - 2*res3_Y
        sigma2_Y_chap = max((1/(n*qY))*np.sum(res_Y_chap))
        
        ####################################################
        #    sigma2_X1_chap
        ####################################################
        for i in range(n):
          dud1_X1[i,] = X1[i,]-np.transpose(D1) @ T1[i,]
        
        res1_X1 = np.apply_along_axis(np.sum, 1, dud1_X1**2)
        res2_X1 = np.array([np.sum(A1**2) * Phi1_tilde])
        tab_res3_X1 = np.zeros((n, 1, n))

        for i in range(n):
          tab_res3_X1[:,:,i] = dud1_X1[i,] @ np.transpose(A1) @ F1_tilde[:,:,i]
        
        res3_X1 = np.apply_over_axes(np.mean, tab_res3_X1, [1,2]).reshape(n)
        #Pour un idividu i, ca revient au meme que  ce qui semble plus logique : t((X[i,]-mu_X[,,it]))%*%(as.matrix(A[,,it]))*F_tilde[,,i]
        res_X1_chap = res1_X1 + res2_X1 - 2*res3_X1
        sigma2_X1_chap = max((1/(n*qX1))*np.sum(res_X1_chap))

        ####################################################
        #    sigma2_X2_chap
        ####################################################

        for i in range(n):
          dud1_X2[i,] = X2[i,]-np.transpose(D2) @ T2[i,]
        
        res1_X2 = np.apply_along_axis(np.sum, 1, dud1_X2**2)
        res2_X2 = np.array([np.sum(A2**2) * Phi2_tilde])
        tab_res3_X2 = np.zeros((n, 1, n))

        for i in range(n):
          tab_res3_X2[:,:,i] = dud1_X2[i,] @ np.transpose(A2) @ F2_tilde[:,:,i]
        
        res3_X2 = np.apply_over_axes(np.mean, tab_res3_X2,[1,2])
        #Pour un idividu i, ca revient au meme que  ce qui semble plus logique : t((X[i,]-mu_X[,,it]))%*%(as.matrix(A[,,it]))*F_tilde[,,i]
        res_X2_chap = res1_X2 + res2_X2 - 2*res3_X2
        sigma2_X2_chap = max((1/(n*qX2))*np.sum(res_X2_chap))
        
        
        
        ########################
        #  Psi_X1, Psi_X2 et Psi_Y a it+1
        ########################
        
        Psi_Y = sigma2_Y_chap * np.eye(qY)
        Psi_X1 = sigma2_X1_chap * np.eye(qX1)
        Psi_X2 = sigma2_X2_chap * np.eye(qX2)

        ##############################
        #---- Conversion des parametres de matrix a numeric----
        #On convertit C1 et C2 en numeric car apres estimation ce sont des matrix et le debut du code est cree avec des C1 et C2 numeric.
        
        C1 = float(C1)
        C2 = float(C2)
        
        #car a l'initialisation ceux sont des numeric et apres application de la boucle while il deviennent des
        #matrix. Si on ne le fait pas, l'application de la boucle lors de la deuxieme fois ne peut etre correcte.
        #Rq : elle n'est pas appliquee aux sigma2 car apres appli de la boucle while leur class n'est pas changee
        
        #mesure du changement d'iteration :
        diff = (np.sum(np.abs(D - D_it) / np.abs(D)) +
            np.sum(np.abs(D1 - D1_it) / np.abs(D1)) +
            np.sum(np.abs(D2 - D2_it) / np.abs(D2)) +
            np.sum(np.abs(B - B_it) / np.abs(B)) +
            np.sum(np.abs(A1 - A1_it) / np.abs(A1)) +
            np.sum(np.abs(A2 - A2_it) / np.abs(A2)) +
            np.abs((C1 - C1_it) / np.abs(C1)) +
            np.abs((C2 - C2_it) / np.abs(C2)) +
            np.abs((sigma2_Y_chap - sigma2_Y_chap_it) / np.abs(sigma2_Y_chap)) +
            np.abs((sigma2_X1_chap - sigma2_X1_chap_it) / np.abs(sigma2_X1_chap)) +
            np.abs((sigma2_X2_chap - sigma2_X2_chap_it) / np.abs(sigma2_X2_chap)))

        
        diffgraph[it]=diff


        print('iteration', it, '\n', '\n',
            'difference', diffgraph[it], 
            '\n','D_chap', D,
            '\n','D1_chap', D1,
            '\n','D2_chap', D2,
            '\n','B_chap', B,
            '\n','A1_chap', A1, '\n','A2_chap', A2,
            '\n','C1_chap', C1, '\n','C2_chap', C2,
            '\n', 'sigma2_Y_chap', sigma2_Y_chap,
            '\n', 'sigma2_X1_chap', sigma2_X1_chap,
            '\n', 'sigma2_X2_chap', sigma2_X2_chap,  '\n'
        )

    #fin boucle while du code

    return {'Factors': np.hstack((G_tilde, F1_tilde, F2_tilde)),
            'C': np.hstack((C1, C2)),
            'B': B,
            'A1': A1,
            'A2': A2,
            'D': D,
            'D1': D1,
            'D2': D2,
            'sigma2': np.array([sigma2_Y_chap, sigma2_X1_chap, sigma2_X2_chap]),
            'Diff': diff,
            'Diff_it': diffgraph,
            'Conv_it': it
          }
  #end function SEM_FM3
  