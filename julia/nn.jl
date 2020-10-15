using RDatasets, Zygote, Plots, Statistics, Base 
using Flux

iris = dataset("datasets", "iris");
num_samples = size(iris)[1];
num_labels  = length(unique(iris[:,:Species]));


X = hcat(Array(iris[:,1:4]), ones(num_samples,1));
Y = Array(iris[:,5]);

function oh_encode(y_orig)
    Y_oh = zeros(Int8, num_samples, num_labels);
    label_to_idx = Dict( label=>idx for (idx,label) in enumerate(unique(Y)));

    for i = 1:num_samples
        Y_oh[i,label_to_idx[Y[i]]] = 1;
    end;

    return Y_oh, label_to_idx
end;

Y_oh, label_to_idx = oh_encode(Y);



mu_1, mu_2 = mean(X[:,1]), mean(X[:,2]);
sig_1, sig_2 = std(X[:,1]), std(X[:,2]);

X[:,1] = (X[:,1] .- mu_1) ./ sig_1;
X[:,2] = (X[:,2] .- mu_2) ./ sig_2;


#W = rand(5,3); # single layer weights
# 5 inputs, 3 classes
b1 = rand(1);
b2 = rand(1);
W1 = rand(5,25);
W2 = rand(25,3);

function fw(X,W1,W2,b1,b2)
    """ make a forward pass """
     return softmax(tanh.(X * W1 .+ b1) * W2 .+ b2, dims=[2])
end;


function ce_loss(y_true, y_pred)
    return - sum( y_true .* log.(y_pred))
end;

# use as softmax(X * W, dims=[2])

loss = [];
acc = [];
Y_true = mapslices(argmax, Y_oh, dims=[2]);
step = 0.001;

for epoch = 1:200
    grads = gradient( () -> ce_loss(Y_oh, fw(X,W1,W2,b1,b2) ), params(W1,W2,b1,b2));

    global W1 -= step .* grads[W1];
    global W2 -= step .* grads[W2];
    global b1 -= step .* grads[b1];
    global b2 -= step .* grads[b2];

    push!(loss, ce_loss(Y_oh, fw(X,W1,W2,b1,b2) ))

    Y_pred = mapslices(argmax, fw(X,W1,W2,b1,b2), dims=[2]);

    push!(acc,sum(Y_true .== Y_pred)/num_samples);
end

# plot(loss)
p1 = plot(acc, title="training acc")
p2 = plot(loss, title="training loss")
plot(p1,p2, layout=(2,1), legend=false)