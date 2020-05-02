functions {
    real[] poisson_ode_fun(real t, real[] y, real [] parms,  real[] x_r, int[] x_i){
        real a = parms[1];
        real b = parms[2];
        real rho = parms[3];
        real dydt[1];

        dydt[1] = - a*y[1]*log(y[1]) - b*(y[1] - y[1]*y[1]) - b*rho*y[1];
        return dydt;
        }
}

data{
    int<lower=0> N; //Number of data points
    real<lower=0> Tmax; //Maximum of infection times
    real<lower=0> infectiontimes[N]; //Ordered infection times
    real<lower=0> t0;
}

transformed data {
    real x_r[0];
    int x_i[0];
}

parameters {
    real<lower=0> a;
    real<lower=0> b;
    real<lower=0, upper=1.0> rho;
}

transformed parameters {
    real R0 = b/a;
    real inv_rho = 1.0/rho;
    real c = b*rho;
}

model {
    real parms[3];
    real ic[1];
    real s[N,1];
    real smax;
    real factor;

    parms[1] = a; parms[2] = b; parms[3] = rho;
    ic[1] = 1.0;

    s = integrate_ode_rk45(poisson_ode_fun,ic,t0,infectiontimes,parms,x_r,x_i);

    smax = s[N,1];
    factor = 1 - smax;

    for (i in 1:N){
        target += log((a*s[i,1]*log(s[i,1]) + b*(s[i,1]-s[i,1]*s[i,1]) + b*rho*s[i,1])/factor);
    }
    target += +gamma_lpdf(a|2,2)+ gamma_lpdf(b|2,2) + beta_lpdf(rho|1,1);
}
