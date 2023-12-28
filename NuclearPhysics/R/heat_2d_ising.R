#------------------------------------------------
# Specifc Heat in 2D Ising Problem
#------------------------------------------------
heat_c2x = function(x){
  out = 4 + 8*x^2 - 8*x^3 - 30*x^4
  #out = x^(-4)*(96 -384/x + 960/x^2 - 1920/x^3 + 2272/x^4)
  return(out)
}


heat_c2x_sg = function(x,ns,ga){
  A = 1.4614
  c = 8.38769
  k = 0:ns
  sk = c*(A^k)*(-1)^(k+1)
  return(ga*sum(sk*x^k))
}

# Function of p,q, sigma1, and sigma2
Sx = function(x,s1,s2,L){
  k = 0:(L-1)
  pq = expand.grid(k,k)
  fpq = cos((2*pq[,1]+s1)*pi/L) + cos((2*pq[,2]+s2)*pi/L)
  if(s1 >0 | s2 >0){
    out = 2^(L^2)*prod((sqrt(cosh(2*x)^2 - sinh(2*x)*fpq)))  
  }else{
    out = (2^(L^2)*prod(sqrt(cosh(2*x)^2 - sinh(2*x)*fpq)))*ifelse(1-sinh(2*x)^2>0,1,-1)
  }
  return(out)
}

# Derivatives of Sx
dx_Sx = function(x,s1,s2,L){
  # Get the product terms 
  k = 0:(L-1)
  pq = expand.grid(k,k)
  fpq = cos((2*pq[,1]+s1)*pi/L) + cos((2*pq[,2]+s2)*pi/L)
  
  # Use the product rule to compute the derivative, store term by term
  Apq = sqrt(cosh(2*x)^2 - sinh(2*x)*fpq)
  Apq_dx = (2*sinh(2*x)*cosh(2*x) - cosh(2*x)*fpq)/sqrt(cosh(2*x)^2 - sinh(2*x)*fpq)
  Apq_dx2 = (2*fpq*sinh(2*x)^2*(fpq - 2*sinh(2*x)) - 
               fpq*cosh(2*x)^2*(fpq + 2*sinh(2*x)) + 
               4*cosh(2*x)^4)/(cosh(2*x)^2 - sinh(2*x)*fpq)^(3/2)
  
  # Get the derivative components (product rule)
  dx1_list = list()
  npq = length(Apq)
  for(i in 1:npq){
    dx1_list[[i]] = Apq
    dx1_list[[i]][i] = Apq_dx[i]
  }
  
  # Get the second derivatives
  dx2_list = list()
  for(i in 1:npq){
    for(j in 1:npq){
      ind = (i-1)*npq + j
      dx2_list[[ind]] = Apq # init
      if(i==j){
        dx2_list[[ind]][j] = Apq_dx2[j] # Second deriv      
      }else{
        dx2_list[[ind]][i] =  Apq_dx[i] # first deriv
        dx2_list[[ind]][j] =  Apq_dx[j] # first deriv
      }
    }
  }
  
  # Get the first and second derivative values
  dx1 = (2^(L^2))*sum(unlist(lapply(dx1_list, prod)))
  dx2 = (2^(L^2))*sum(unlist(lapply(dx2_list, prod)))
  
  if(s1 == 0 & s2 == 0){
    #dx1 = dx1*ifelse(1-sinh(2*x)^2>0,1,-1) - (2^(L^2))*prod(Apq)*(4*cosh(2*x)*sinh(2*x))
    #dx2 = dx2*ifelse(1-sinh(2*x)^2>0,1,-1) - (2^(L^2))*dx1*(4*cosh(2*x)*sinh(2*x)) -
    #  (2^(L^2))*dx1*(4*cosh(2*x)*sinh(2*x)) - (2^(L^2))*prod(Apq)*(8*cosh(2*x)^2 + 8*sinh(2*x)^2) 
    
    dx1 = dx1*ifelse(1-sinh(2*x)^2>0,1,-1)
    dx2 = dx2*ifelse(1-sinh(2*x)^2>0,1,-1)
  }
  
  out = list(dx1 = dx1, dx2 = dx2)
  return(out)
}

# Zx function - uses the Sx function
Zx = function(x,L){
  # Get the S terms for different sig1 and sig2
  S11 = Sx(x,1,1,L)
  S10 = Sx(x,1,0,L)
  S00 = Sx(x,0,0,L)
  # Get the z value
  z = 0.5*(S11 + 2*S10 - S00)
  return(z)
}


# Second derivative of the parition function to get specific heat
# Use finite differences for deriv wrt x and then put in terms of g
# exp(2x) = g + 1
specific_heat = function(g,L){
  # Convert from g to x
  x = 0.5*log(g+1)
  
  # Compute the Sij(X) values and derivatives
  s11 = Sx(x,1,1,L)
  s11_prime = dx_Sx(x,1,1,L)
  
  s10 = Sx(x,1,0,L)
  s10_prime = dx_Sx(x,1,0,L)

  s00 = Sx(x,0,0,L)
  s00_prime = dx_Sx(x,0,0,L)

  # Compute Z(x) and derivatives
  z = Zx(x,L)
  zdx = 0.5*(s11_prime$dx1 + 2*s10_prime$dx1 - s00_prime$dx1)
  zdx2 = 0.5*(s11_prime$dx2 + 2*s10_prime$dx2 - s00_prime$dx2)
  
  dx2_logz = (zdx2*z - zdx^2)/z^2
  clx = dx2_logz/L^2
  return(clx)
}

# Second order derivative approx
d2_logz = function(x,L,h){
  lz1 = log(Zx(x+h,L))
  lz2 = log(Zx(x-h,L))
  lz = log(Zx(x,L))
  return((lz1 + lz2 - 2*lz)/h^2)
}




# TEST
gvec = seq(0.01,5,length=200)
xvec = 0.5*log(gvec+1)
sh = sapply(gvec, function(g) specific_heat(g,2))
plot(gvec,sh, ylim = c(0,6), type='l')


s00_prime = sapply(xvec, function(x) dx_Sx(x,0,0,2)$dx2)
s10_prime = sapply(xvec, function(x) dx_Sx(x,1,0,2)$dx1)
s11_prime = sapply(xvec, function(x) dx_Sx(x,1,1,2)$dx1)
s00 = sapply(xvec, function(x) Sx(x,0,0,2))
s11 = sapply(xvec, function(x) Sx(x,1,1,2))
plot(gvec,s00_prime, type = "l", ylim = c(-2000,1))
plot(gvec,s10_prime, type = "l")
plot(gvec,s11_prime, type = "l")
plot(gvec,s00, type = "l")
plot(gvec,s11, type = "l")



L = 5
dz2 = sapply(xvec,function(x) d2_logz(x,L,0.1))
plot(gvec,dz2/L^2, type = 'l')
