using GaussianProcesses, Random, RDatasets

Random.seed!(113355)

crabs = dataset("MASS", "crabs");
crabs = crabs[shuffle(1:size(crabs)[1]), :] # shuffle dataset

train = crabs[1:div(end,2), :];

y = Array{Bool}(undef, size(train)[1]);
y[train[:Sp] .== "B"] .= 0;
y[train[:Sp] .== "O"] .= 1;

X = convert(Array, train[:,4:end]);

# select mean, kernel, likelihood functions
mZero = MeanZero();
kern = Matern(3/2, zeros(5), 0.0);
lik = BernLik();

# fit the GP model using the general GP function (GPMC)
gp = GP(X', y, mZero, kern, lik)


# perform sampling
#samples = mcmc(gp, ϵ=0.01, nIter=10000, burn=1000, thin=10);

