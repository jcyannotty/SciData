#------------------------------------------------
# EFT Pade and FPR Interpolation between two expansions
#------------------------------------------------
rx = function(x, avec, bvec){
  na = length(avec)
  nb = length(bvec)
  A = sum(avec*x^((1:na)-1))
  B = sum(1 + bvec*x^(1:nb))
  return(A/B)
} 

# kth order derivative of polynomial
dp_dx = function(x,cvec,k){
  nc = length(cvec)
  power_vec = 0:(nc-1)

  if(k>0){
    for(i in 1:k){
      cvec = cvec*power_vec
      power_vec = sapply(power_vec, function(x) max(x-i,0))
    }
  }
  
  out = list(value = sum(cvec*x^(power_vec)), coefs = cvec, powers = power_vec)
  return(out)
}

# kth order derivative of r(x)
drk_dx = function(x,avec,bvec,k){
  # Add the 1 to bvec and get sizes
  bvec = c(1,bvec)
  nb = length(bvec)
  na = length(avec)
  
  # Define polynomials
  Bx = sum(bvec*x^(0:(nb-1)))
  Ax = sum(avec*x^(0:(na-1)))
  
  rkx = (dp_dx(x,avec,k)$value*Bx - Ax*dp_dx(x,bvec,k)$value)/(Bx^2)
  return(rkx)
}

pade_constraints = function(cvec0,cvec1,x0,x1,m0,m1){
  # Build constraints for first ts centered at x0 and x1 respectively
  rx_values = 0
  for(i in 0:m0){
    rx_values[i+1] = dp_dx(0,cvec0,i)$value
  }
  
  for(i in 0:m1){
    rx_values[m0+i+2] = dp_dx(0,cvec1,i)$value
  }
  return(rx_values)
}


# Requires global parameters to be defined: avec, bvec, x0, x1, m1, m2
pade_loss = function(cvec){
  na = length(avec) 
  nb = length(bvec)
  avec = cvec[1:na]
  bvec = cvec[(na+1):(na+nb)]
  
  # Left hand side constraints centered about x0
  crx_values = 0
  for(i in 0:m0){
    if(i == 0){
      crx_values[i+1] = rx(x0,avec,bvec)
    }else{
      crx_values[i+1] = drk_dx(x0,avec,bvec,i)
    }
  }
  
  # Left hand side constraints centered about x1
  for(i in 0:m1){
    if(i == 0){
      crx_values[m0+i+2] = rx(x1,avec,bvec)
    }else{
      crx_values[m0+i+2] = drk_dx(x1,avec,bvec,i)
    }
  }
  
  # Squared error loss
  return(sum((rx_values - crx_values)^2))
}


hcs = heat_sg_exp(1,10,8,TRUE)
hcl = heat_lg_exp(1,10,8,TRUE)

m0 = 3
m1 = 3
avec = hcs$coefs[1:(m0+1),2]
bvec = hcl$coefs[1:(m1+1),2]
x0 = 0
x1 = 4
rx_values = pade_constraints(hcs$coefs[,2],hcl$coefs[,2],x0,x1,m0,m1)

op_out = optim(c(avec,rep(0,length(bvec))), pade_loss)

# Get the resuls
avec_star = op_out$par[1:(m0+1)]
bvec_star = op_out$par[(m0+2):(m0+m1+2)]

pade_out = sapply(gvec, function(g) rx(g,avec_star,bvec_star))
plot(gvec,pade_out)


pade_loss(op_out$par)
